"""Futures strategy base class for CZCE multi-product backtests."""

from datetime import date, datetime
import os
import sys

import backtrader as bt

_BACKTEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKTEST_DIR)

from products import normalize_symbol, parse_product


def _to_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, 'date') and callable(value.date):
        return value.date()
    return value


class ProductFeeds:
    """Weighted series plus real-contract feeds for one product."""

    __slots__ = ('symbol', 'weighted', 'contracts')

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.weighted = None
        self.contracts = {}


class FuturesStrategyBase(bt.Strategy):
    """
    Futures strategy base class.

    Provides common helpers such as signal logging and trade logging.
    Subclasses implement the actual logic in ``next()``.

    The backtest engine injects the following params via cerebro.addstrategy():
      - symbol              : default product for self.data / unmarked orders
      - symbols             : products loaded into this run
      - execute_on_contracts: route orders to calendar contracts (default False)
      - contract_by_date    : {symbol: {date: contract_code}}
      - first_print         : {symbol: date} listing dates
      - trade_size          : lots per trade (default 1)
      - margin_rate         : margin ratio

    Feeds:
      self.data / self.weighted     default product's OI-weighted series (datas[0])
      self.contracts[code]          that product's real contracts
      self.products[sym].weighted   any loaded product's weighted series
      self.products[sym].contracts  that product's real contracts

    When ``execute_on_contracts`` is True, buy/sell/close go to the calendar
    contract (unless you pass ``data=``) and rolls happen in ``next_open``.

    Subclasses can directly use:
      self.buy_signal()                 open long on the default product
      self.buy_signal(symbol='FG')      open long on glass
      self.sell_signal(symbol='CF')     open short on cotton
      self.close_signal(symbol='SA')    close one product
      self.get_position_size(symbol=)   net lots for one product
    """

    params = (
        ('contract_multiplier', 20),
        ('trade_size', 1),
        ('margin_rate', 0.10),
        ('printlog', False),
        ('execute_on_contracts', False),
        ('contract_by_date', None),
        ('symbol', 'SA'),
        ('symbols', None),
        ('first_print', None),
    )

    def __init__(self):
        self.signal_log = []
        self._pending_by_symbol = {}
        self._pending_refs = set()
        self._pending_ref_symbol = {}
        self._exec_data_by_symbol = {}
        self._exec_code_by_symbol = {}
        self._roll_order_refs = set()
        self._contract_by_date = {}
        self._roll_done_on = {}
        self._protect_stop_by_symbol = {}
        self._stop_distance_by_symbol = {}
        self._stop_price_by_symbol = {}
        self._first_print = {}

        self.products = {}
        discovered = []
        for feed in self.datas:
            name = getattr(feed, '_name', '') or ''
            if not name:
                continue
            if name.lower().endswith('_weighted'):
                sym = normalize_symbol(name[:-len('_weighted')])
                bucket = self.products.setdefault(sym, ProductFeeds(sym))
                bucket.weighted = feed
                if sym not in discovered:
                    discovered.append(sym)
                continue
            sym = parse_product(name)
            if not sym:
                continue
            bucket = self.products.setdefault(sym, ProductFeeds(sym))
            bucket.contracts[name] = feed
            if sym not in discovered:
                discovered.append(sym)

        param_symbols = [
            normalize_symbol(s) for s in (self.p.symbols or []) if s
        ]
        self.symbols = param_symbols or discovered
        self._default_symbol = normalize_symbol(
            self.p.symbol or (self.symbols[0] if self.symbols else 'SA')
        )

        default_bucket = self.products.get(self._default_symbol)
        if default_bucket and default_bucket.weighted is not None:
            self.weighted = default_bucket.weighted
        else:
            self.weighted = self.datas[0]
        self.contracts = default_bucket.contracts if default_bucket else {}

        if self._default_symbol not in self.products:
            loaded = ', '.join(self.symbols) or '(none)'
            raise ValueError(
                f"Strategy symbol {self._default_symbol!r} is not among "
                f"loaded product feeds: {loaded}"
            )

    @property
    def _pending_order(self):
        return self._pending_by_symbol.get(self._default_symbol)

    @_pending_order.setter
    def _pending_order(self, value):
        self._pending_by_symbol[self._default_symbol] = value

    @property
    def _exec_data(self):
        return self._exec_data_by_symbol.get(self._default_symbol)

    @property
    def _exec_code(self):
        return self._exec_code_by_symbol.get(self._default_symbol)

    @property
    def _protect_stop_order(self):
        return self._protect_stop_by_symbol.get(self._default_symbol)

    @_protect_stop_order.setter
    def _protect_stop_order(self, value):
        if value is None:
            self._protect_stop_by_symbol.pop(self._default_symbol, None)
        else:
            self._protect_stop_by_symbol[self._default_symbol] = value

    @property
    def _stop_distance(self):
        return self._stop_distance_by_symbol.get(self._default_symbol)

    @_stop_distance.setter
    def _stop_distance(self, value):
        if value is None:
            self._stop_distance_by_symbol.pop(self._default_symbol, None)
        else:
            self._stop_distance_by_symbol[self._default_symbol] = value

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def start(self):
        raw = self.p.contract_by_date or {}
        self._contract_by_date = self._normalize_roll_map(raw)
        self._first_print = {
            normalize_symbol(k): _to_date(v)
            for k, v in (self.p.first_print or {}).items()
        }
        if self.p.execute_on_contracts:
            self._maybe_roll()

    def prenext_open(self):
        if self.p.execute_on_contracts:
            self._maybe_roll()

    def next_open(self):
        if self.p.execute_on_contracts:
            self._maybe_roll()

    def stop(self):
        leftovers = {}
        for symbol in self.symbols:
            target = self._exec_code_by_symbol.get(symbol)
            for feed, sz in self._positions_by_data(symbol).items():
                name = getattr(feed, '_name', '')
                if name != target:
                    leftovers[name] = sz
        if leftovers:
            print(f"  [WARN] Leftover positions on non-calendar contracts: {leftovers}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        is_roll = self._is_roll_order(order)
        self._roll_order_refs.discard(order.ref)
        symbol = self._symbol_of_order(order)

        date = self.datas[0].datetime.date(0)

        if order.status == order.Completed:
            if self._is_protect_stop(order):
                if self._protect_stop_by_symbol.get(symbol) is order:
                    self._protect_stop_by_symbol.pop(symbol, None)
                self._stop_distance_by_symbol.pop(symbol, None)
            elif is_roll and self._stop_distance_by_symbol.get(symbol):
                psz = int(self.getposition(order.data).size)
                if psz:
                    self.arm_protect_stop(
                        order.executed.price,
                        is_long=psz > 0,
                        size=abs(psz),
                        data=order.data,
                        symbol=symbol,
                    )
            if not is_roll:
                direction = 'buy' if order.isbuy() else 'sell'
                self.signal_log.append({
                    'date': date,
                    'price': order.executed.price,
                    'direction': direction,
                    'size': order.executed.size,
                    'comm': order.executed.comm,
                    'symbol': symbol,
                })
            if self.p.printlog:
                tag = 'ROLL ' if is_roll else ''
                cname = getattr(order.data, '_name', '') or ''
                print(
                    f"  [{date}] {tag}Filled: {'BUY' if order.isbuy() else 'SELL'} "
                    f"{symbol} {cname} price={order.executed.price:.2f} "
                    f"size={order.executed.size} "
                    f"commission={order.executed.comm:.2f}"
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self._is_protect_stop(order):
                if self._protect_stop_by_symbol.get(symbol) is order:
                    self._protect_stop_by_symbol.pop(symbol, None)
            if self.p.printlog:
                print(f"  [{date}] Order not filled: {order.getstatusname()}")

        self._clear_pending_ref(order.ref)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        if self.p.printlog:
            date = self.datas[0].datetime.date(0)
            print(f"  [{date}] Trade closed: pnl={trade.pnl:.2f}  net_pnl={trade.pnlcomm:.2f}")

    # ------------------------------------------------------------------
    # Order routing (weighted = signals, calendar contract = execution)
    # ------------------------------------------------------------------

    def buy(self, data=None, symbol=None, **kwargs):
        data = self._resolve_exec_data(data, symbol=symbol)
        if self.p.execute_on_contracts and data is None:
            return None
        order = super().buy(data=data, **kwargs)
        self._tag_order(order, symbol or self._symbol_of_data(data))
        return order

    def sell(self, data=None, symbol=None, **kwargs):
        data = self._resolve_exec_data(data, symbol=symbol)
        if self.p.execute_on_contracts and data is None:
            return None
        order = super().sell(data=data, **kwargs)
        self._tag_order(order, symbol or self._symbol_of_data(data))
        return order

    def close(self, data=None, symbol=None, **kwargs):
        if data is not None:
            order = super().close(data=data, **kwargs)
            self._tag_order(order, symbol or self._symbol_of_data(data))
            return order
        symbol = self._resolve_symbol(symbol)
        if not self.p.execute_on_contracts:
            data = self._weighted(symbol)
            order = super().close(data=data, **kwargs)
            self._tag_order(order, symbol)
            return order
        self._sync_exec_pointer(symbol)
        last = None
        for feed in self._contract_datas(symbol):
            if self.getposition(feed).size:
                last = super().close(data=feed, **kwargs)
                self._tag_order(last, symbol)
        return last

    def _resolve_exec_data(self, data, symbol=None):
        if data is not None:
            return data
        symbol = self._resolve_symbol(symbol)
        if self.p.execute_on_contracts:
            if not self._is_listed(symbol):
                return None
            self._sync_exec_pointer(symbol)
            return self._exec_data_by_symbol.get(symbol)
        return self._weighted(symbol)

    def _today(self):
        try:
            return self.datas[0].datetime.date(0)
        except Exception:
            return None

    def _resolve_symbol(self, symbol=None):
        if symbol:
            return normalize_symbol(symbol)
        return self._default_symbol

    def _weighted(self, symbol=None):
        symbol = self._resolve_symbol(symbol)
        bucket = self.products.get(symbol)
        if bucket and bucket.weighted is not None:
            return bucket.weighted
        if symbol == self._default_symbol:
            return self.datas[0]
        return None

    def get_weighted(self, symbol=None):
        """Return the OI-weighted feed for ``symbol`` (default product if omitted)."""
        return self._weighted(symbol)

    def _is_listed(self, symbol=None, dt=None):
        symbol = self._resolve_symbol(symbol)
        dt = _to_date(dt or self._today())
        first = self._first_print.get(symbol)
        if first is None or dt is None:
            return True
        return dt >= first

    def _normalize_roll_map(self, raw):
        if not raw:
            return {}
        values = list(raw.values())
        if values and all(isinstance(v, dict) for v in values):
            return {
                normalize_symbol(sym): {_to_date(k): code for k, code in mapping.items()}
                for sym, mapping in raw.items()
            }
        return {
            self._default_symbol: {_to_date(k): code for k, code in raw.items()}
        }

    def _target_code(self, symbol=None, dt=None):
        symbol = self._resolve_symbol(symbol)
        dt = _to_date(dt or self._today())
        if dt is None:
            return None
        return self._contract_by_date.get(symbol, {}).get(dt)

    def _data_by_code(self, code):
        if not code:
            return None
        try:
            return self.getdatabyname(code)
        except Exception:
            return None

    def _contract_datas(self, symbol=None):
        symbol = self._resolve_symbol(symbol)
        bucket = self.products.get(symbol)
        if not bucket:
            return []
        return list(bucket.contracts.values())

    def _symbol_of_data(self, data):
        if data is None:
            return self._default_symbol
        name = getattr(data, '_name', '') or ''
        parsed = parse_product(name)
        return parsed or self._default_symbol

    def _symbol_of_order(self, order):
        info = getattr(order, 'info', None)
        if info is not None:
            try:
                tagged = info.get('symbol')
                if tagged:
                    return normalize_symbol(tagged)
            except Exception:
                tagged = getattr(info, 'symbol', None)
                if tagged:
                    return normalize_symbol(tagged)
        return self._symbol_of_data(getattr(order, 'data', None))

    def _tag_order(self, order, symbol):
        if order is None or not symbol:
            return
        addinfo = getattr(order, 'addinfo', None)
        if callable(addinfo):
            order.addinfo(symbol=normalize_symbol(symbol))

    def _sync_exec_pointer(self, symbol=None):
        """Point execution at today's calendar contract without placing orders."""
        if not self.p.execute_on_contracts:
            return
        symbol = self._resolve_symbol(symbol)
        code = self._target_code(symbol)
        data = self._data_by_code(code)
        if data is None:
            return
        self._exec_data_by_symbol[symbol] = data
        self._exec_code_by_symbol[symbol] = code

    def _is_protect_stop(self, order):
        if order is None:
            return False
        if order in self._protect_stop_by_symbol.values():
            return True
        info = getattr(order, 'info', None)
        if info is None:
            return False
        try:
            return bool(info.get('is_protect_stop', False))
        except Exception:
            return bool(getattr(info, 'is_protect_stop', False))

    def cancel_protect_stop(self, symbol=None, data=None):
        if data is not None and symbol is None:
            symbol = self._symbol_of_data(data)
        symbol = self._resolve_symbol(symbol)
        order = self._protect_stop_by_symbol.pop(symbol, None)
        if order is None:
            return
        if getattr(order, 'alive', lambda: False)():
            self._cancel_broker_order(order)

    def arm_protect_stop(self, fill_price, is_long, size, data=None,
                         distance=None, symbol=None):
        symbol = self._resolve_symbol(symbol or self._symbol_of_data(data))
        dist = self._stop_distance_by_symbol.get(symbol) if distance is None else distance
        if not dist or size <= 0:
            return None
        self._stop_distance_by_symbol[symbol] = float(dist)
        stop_price = float(fill_price) - dist if is_long else float(fill_price) + dist
        return self.place_protect_stop(
            stop_price, size=int(size), data=data, symbol=symbol
        )

    def place_protect_stop(self, stop_price, size, data=None, symbol=None):
        """Resting stop on the traded contract. Does not block next()."""
        symbol = self._resolve_symbol(symbol or self._symbol_of_data(data))
        self.cancel_protect_stop(symbol=symbol)
        data = self._resolve_exec_data(data, symbol=symbol)
        if self.p.execute_on_contracts and data is None:
            return None
        stop_price = float(stop_price)
        size = abs(int(size))
        if size <= 0:
            return None
        pos = int(self.getposition(data).size) if data is not None else int(self.position.size)
        if pos > 0:
            order = super().sell(
                data=data, size=size, exectype=bt.Order.Stop, price=stop_price
            )
        elif pos < 0:
            order = super().buy(
                data=data, size=size, exectype=bt.Order.Stop, price=stop_price
            )
        else:
            return None
        if order is None:
            return None
        self._mark_protect_stop(order)
        self._tag_order(order, symbol)
        self._protect_stop_by_symbol[symbol] = order
        self._stop_price_by_symbol[symbol] = stop_price
        return order

    def _mark_protect_stop(self, order):
        if order is None:
            return
        addinfo = getattr(order, 'addinfo', None)
        if callable(addinfo):
            order.addinfo(is_protect_stop=True)

    def _fill_protect_stop_if_touched(self, symbol=None):
        """If today's contract range already pierced the resting stop, fill now.

        Backtrader only evaluates a newly submitted Stop on the next broker
        cycle. A cloud-hosted stop would be live on the fill bar, so this
        catches same-bar touches at the stop (or the open if it gapped).
        """
        symbols = [self._resolve_symbol(symbol)] if symbol else list(self.symbols)
        filled_any = False
        for sym in symbols:
            if self._fill_protect_stop_symbol(sym):
                filled_any = True
        return filled_any

    def _fill_protect_stop_symbol(self, symbol):
        order = self._protect_stop_by_symbol.get(symbol)
        if order is None or not getattr(order, 'alive', lambda: False)():
            if order is not None and not getattr(order, 'alive', lambda: False)():
                self._protect_stop_by_symbol.pop(symbol, None)
            return False
        data = getattr(order, 'data', None)
        if data is None:
            return False
        try:
            popen = float(data.open[0])
            phigh = float(data.high[0])
            plow = float(data.low[0])
            pclose = float(data.close[0])
            stop = float(order.created.price)
        except Exception:
            return False
        if order.issell():
            hit = popen <= stop or plow <= stop
        else:
            hit = popen >= stop or phigh >= stop
        if not hit:
            return False
        broker = self.broker
        try_exec = getattr(broker, '_try_exec_stop', None)
        if not callable(try_exec):
            return False
        try_exec(order, popen, phigh, plow, stop, pclose)
        if getattr(order, 'alive', lambda: True)():
            return False
        for attr in ('pending', 'submitted'):
            queue = getattr(broker, attr, None)
            if not queue:
                continue
            try:
                queue.remove(order)
            except ValueError:
                pass
        self._protect_stop_by_symbol.pop(symbol, None)
        return True

    def _is_roll_order(self, order):
        if order is None:
            return False
        if order.ref in self._roll_order_refs:
            return True
        info = getattr(order, 'info', None)
        if info is None:
            return False
        try:
            return bool(info.get('is_roll', False))
        except Exception:
            return bool(getattr(info, 'is_roll', False))

    def _mark_roll(self, order):
        if order is None:
            return
        self._roll_order_refs.add(order.ref)
        addinfo = getattr(order, 'addinfo', None)
        if callable(addinfo):
            order.addinfo(is_roll=True)

    def _track_pending(self, order, symbol=None):
        if order is None:
            return
        if symbol is None:
            symbol = self._symbol_of_data(getattr(order, 'data', None))
        symbol = self._resolve_symbol(symbol)
        self._pending_by_symbol[symbol] = order
        self._pending_refs.add(order.ref)
        self._pending_ref_symbol[order.ref] = symbol

    def _clear_pending_ref(self, ref):
        self._pending_refs.discard(ref)
        symbol = self._pending_ref_symbol.pop(ref, None)
        if symbol is None:
            return
        still = [r for r, s in self._pending_ref_symbol.items() if s == symbol]
        current = self._pending_by_symbol.get(symbol)
        if not still:
            self._pending_by_symbol[symbol] = None
        elif current is not None and getattr(current, 'ref', None) == ref:
            self._pending_by_symbol[symbol] = True

    def has_pending(self, symbol=None):
        """True if the given product (default: strategy symbol) has a live order."""
        return bool(self._pending_by_symbol.get(self._resolve_symbol(symbol)))

    def _order_signed_size(self, order):
        raw = getattr(order, 'created', None)
        size = abs(int(getattr(raw, 'size', None) or order.size or 0))
        if not size:
            return 0
        return size if order.isbuy() else -size

    def _alive_broker_orders(self):
        orders = []
        seen = set()
        for attr in ('submitted', 'pending'):
            queue = getattr(self.broker, attr, None)
            if not queue:
                continue
            for order in list(queue):
                if order is None or id(order) in seen:
                    continue
                seen.add(id(order))
                orders.append(order)
        return orders

    def _cancel_broker_order(self, order):
        """Cancel an order in ``submitted`` or ``pending``.

        Backtrader's ``broker.cancel`` only searches ``pending``. Signal
        orders from the previous ``next()`` are still in ``submitted``
        when ``next_open`` runs, so they must be removed from that queue.
        """
        if order is None:
            return False
        broker = self.broker
        submitted = getattr(broker, 'submitted', None)
        if submitted is not None:
            try:
                submitted.remove(order)
            except ValueError:
                pass
            else:
                order.cancel()
                broker.notify(order)
                return True
        return bool(broker.cancel(order))

    def _positions_by_data(self, symbol=None):
        held = {}
        for feed in self._contract_datas(symbol):
            size = int(self.getposition(feed).size)
            if size:
                held[feed] = size
        return held

    def _maybe_roll(self):
        """Keep each product's exposure on today's calendar contract."""
        if not self.p.execute_on_contracts:
            return
        today = self._today()
        if today is None:
            return
        for symbol in self.symbols:
            self._maybe_roll_symbol(symbol, today)

    def _maybe_roll_symbol(self, symbol, today):
        if self._roll_done_on.get(symbol) == today:
            return
        if not self._is_listed(symbol, today):
            return
        self._roll_done_on[symbol] = today

        new_code = self._target_code(symbol, today)
        if not new_code:
            return

        new_data = self._data_by_code(new_code)
        if new_data is None:
            if self.p.printlog:
                print(
                    f"  [{today}] No feed for {symbol} {new_code}, "
                    f"keeping {self._exec_code_by_symbol.get(symbol)}"
                )
            return

        held = self._positions_by_data(symbol)
        pos_target = int(self.getposition(new_data).size)
        pos_others = sum(sz for feed, sz in held.items() if feed is not new_data)

        wrong_pending = 0
        for order in self._alive_broker_orders():
            if self._is_roll_order(order):
                continue
            if order.status not in (order.Submitted, order.Accepted, order.Partial):
                continue
            data = getattr(order, 'data', None)
            if data is None or self._symbol_of_data(data) != symbol:
                continue
            if data is self._weighted(symbol):
                continue
            if data is new_data:
                continue
            if self._is_protect_stop(order):
                self._cancel_broker_order(order)
                if self._protect_stop_by_symbol.get(symbol) is order:
                    self._protect_stop_by_symbol.pop(symbol, None)
                continue
            wrong_pending += self._order_signed_size(order)
            self._cancel_broker_order(order)

        move = pos_others + wrong_pending

        old_code = self._exec_code_by_symbol.get(symbol)
        self._exec_data_by_symbol[symbol] = new_data
        self._exec_code_by_symbol[symbol] = new_code

        if not pos_others and not wrong_pending:
            return

        for feed, size in held.items():
            if feed is new_data or not size:
                continue
            close_ord = super().close(data=feed)
            self._mark_roll(close_ord)
            self._tag_order(close_ord, symbol)

        open_ord = None
        if move > 0:
            open_ord = super().buy(data=new_data, size=move)
        elif move < 0:
            open_ord = super().sell(data=new_data, size=abs(move))
        self._tag_order(open_ord, symbol)

        if pos_others:
            self._mark_roll(open_ord)
        elif open_ord is not None:
            self._track_pending(open_ord, symbol)

        if self.p.printlog:
            print(
                f"  [{today}] Roll {symbol} {old_code} -> {new_code} "
                f"others={pos_others} pending={wrong_pending} move={move} "
                f"on_target={pos_target}"
            )

    # ------------------------------------------------------------------
    # Utility methods (for subclasses)
    # ------------------------------------------------------------------

    def buy_signal(self, size=None, data=None, symbol=None):
        """Open long at market (futures long entry)."""
        symbol = self._resolve_symbol(symbol or self._symbol_of_data(data))
        if not self._is_listed(symbol):
            return
        if self.has_pending(symbol):
            return
        self._track_pending(
            self.buy(
                data=data,
                symbol=symbol,
                size=size if size is not None else self.p.trade_size,
            ),
            symbol,
        )

    def sell_signal(self, size=None, data=None, symbol=None):
        """Open short at market (futures short entry)."""
        symbol = self._resolve_symbol(symbol or self._symbol_of_data(data))
        if not self._is_listed(symbol):
            return
        if self.has_pending(symbol):
            return
        self._track_pending(
            self.sell(
                data=data,
                symbol=symbol,
                size=size if size is not None else self.p.trade_size,
            ),
            symbol,
        )

    def close_signal(self, data=None, symbol=None):
        """Close the current position on one product (default: strategy symbol)."""
        if data is not None:
            symbol = self._resolve_symbol(symbol or self._symbol_of_data(data))
        else:
            symbol = self._resolve_symbol(symbol)
        self.cancel_protect_stop(symbol=symbol)
        if self.has_pending(symbol):
            return
        if data is not None:
            self._track_pending(self.close(data=data, symbol=symbol), symbol)
            return
        if self.p.execute_on_contracts:
            last = None
            for feed in self._contract_datas(symbol):
                if self.getposition(feed).size:
                    last = super().close(data=feed)
                    self._tag_order(last, symbol)
                    self._track_pending(last, symbol)
            if last is None:
                self._track_pending(self.close(symbol=symbol), symbol)
            return
        self._track_pending(self.close(symbol=symbol), symbol)

    def get_position_size(self, symbol=None) -> int:
        """Return net lots for one product (positive=long, negative=short, 0=flat)."""
        symbol = self._resolve_symbol(symbol)
        if self.p.execute_on_contracts:
            feeds = self._contract_datas(symbol)
            if feeds:
                return int(sum(self.getposition(d).size for d in feeds))
            data = self._exec_data_by_symbol.get(symbol)
            if data is not None:
                return int(self.getposition(data).size)
            return 0
        data = self._weighted(symbol)
        if data is None:
            return 0
        return int(self.getposition(data).size)

    def get_contract(self, code: str):
        """Return the real-contract feed for ``code``, or None if not loaded."""
        if not code:
            return None
        for bucket in self.products.values():
            found = bucket.contracts.get(code)
            if found is not None:
                return found
        if code in self.contracts:
            return self.contracts[code]
        return self._data_by_code(code)
