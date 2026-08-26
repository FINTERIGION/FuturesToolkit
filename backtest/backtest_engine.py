"""
Backtest Engine Module
Responsible for:
  - Custom analyzers (equity curve, daily returns, position records)
  - Trade log collection and CSV export
  - Metrics calculation (Sharpe, max drawdown, recovery period, win rate, etc.)
  - Running Cerebro and returning structured results
"""

import os
import collections
import math
import datetime
import csv
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np

from .products import normalize_symbol, parse_product, product_costs, weighted_feed_name


# ==============================================================================
# Futures cost model and custom analyzers
# ==============================================================================

class FuturesCommissionInfo(bt.CommissionInfo):
    """Commission model for percentage-margined futures contracts.

    Backtrader's ``margin`` parameter is an absolute currency amount per lot,
    not a margin ratio. ``get_margin`` keeps the required margin dynamic as
    the price changes.

    Fee is either a fraction of notional (``per_lot=False``) or a fixed
    CNY amount per lot (``per_lot=True``).
    """

    params = (
        ('margin_rate', 0.10),
        ('per_lot', False),
    )

    def get_margin(self, price):
        """Return margin per lot: price × multiplier × margin rate."""
        return price * self.p.mult * self.p.margin_rate

    def getvalue(self, position, price):
        """Equity contribution of a position: margin locked at the entry price.

        Backtrader takes ``getoperationcost(size, entry)`` out of cash when a
        position opens and adds ``getvalue(position, close)`` back when pricing
        the account. With a price-dependent ``get_margin`` those two do not
        cancel, and the leftover term rides on the equity curve:

            equity = true_equity + |size| * mult * margin_rate * (close - entry)

        i.e. longs were reported at ``true_pnl * (1 + margin_rate)`` and shorts
        at ``true_pnl * (1 - margin_rate)`` - an 11-13% error on these products,
        in opposite directions. Valuing at ``position.price`` makes the pair
        cancel exactly, so only the mark-to-market P&L that ``cashadjust``
        already booked into cash moves the equity.

        Ignoring ``price`` also keeps flat positions from poisoning the account
        value. ``Strategy.getposition`` seeds ``broker.positions`` with a
        zero-size entry for every feed it is asked about, and a contract that
        has not printed yet carries ``close = nan``: ``abs(0) * get_margin(nan)``
        is nan, which is enough to turn the whole equity curve into nan.

        Requires ``broker.set_shortcash(False)`` - see ``_build_cerebro``.
        """
        return abs(position.size) * self.get_margin(position.price)

    def profitandloss(self, size, price, newprice):
        """Same nan guard for the unrealized leg of ``BackBroker._get_value``."""
        if not size:
            return 0.0
        return super().profitandloss(size, price, newprice)

    def _getcommission(self, size, price, pseudoexec):
        lots = abs(size)
        if self.p.per_lot:
            return lots * self.p.commission
        return lots * price * self.p.mult * self.p.commission


class DailyEquityAnalyzer(bt.Analyzer):
    """
    Records, for every bar: date, total account equity, per-product net
    position, and daily return.

    ``position`` is a ``{symbol: net_lots}`` dict covering every product
    loaded into the run, not just the strategy's default symbol - a
    multi-product strategy can be long one product and short another on the
    same day, and collapsing that to a single scalar hid every product but
    the default one from the position/summary charts.
    """

    def start(self):
        self.equity_records = []      # [{date, equity, position, daily_return}]
        self._prev_equity = self.strategy.broker.getvalue()
        self._symbols = list(getattr(self.strategy, 'symbols', None) or [])

    def next(self):
        dt = self.strategy.datas[0].datetime.date(0)
        equity = self.strategy.broker.getvalue()
        pos = self._positions()
        daily_return = (equity - self._prev_equity) / self._prev_equity if self._prev_equity else 0.0
        self.equity_records.append({
            'date': dt,
            'equity': equity,
            'position': pos,
            'daily_return': daily_return,
        })
        self._prev_equity = equity

    def _positions(self):
        strat = self.strategy
        getter = getattr(strat, 'get_position_size', None)
        if callable(getter) and self._symbols:
            out = {}
            for symbol in self._symbols:
                try:
                    out[symbol] = getter(symbol)
                except TypeError:
                    out[symbol] = getter()
            return out
        # Fallback for a bare backtrader Strategy without the multi-product
        # helpers from FuturesStrategyBase.
        if getattr(strat.p, 'execute_on_contracts', False) and len(strat.datas) > 1:
            net = sum(strat.getposition(d).size for d in strat.datas[1:])
        else:
            net = strat.position.size
        symbol = getattr(strat.p, 'symbol', None) or 'default'
        return {symbol: net}

    def get_analysis(self):
        return self.equity_records


TRADE_LOG_FIELDS = [
    'trade_id', 'open_date', 'close_date', 'direction', 'symbol',
    'contract', 'contracts', 'n_rolls',
    'open_price', 'close_price', 'size',
    'gross_pnl', 'commission', 'net_pnl', 'margin_used', 'open_at_end',
]


class TradeLogAnalyzer(bt.Analyzer):
    """
    One row per *logical* trade: the signal entry that takes a product off flat
    through the signal exit that flattens it again, with every calendar roll in
    between folded into that same row.

    Two reasons this is a fill-driven ledger rather than a wrapper around
    backtrader's ``Trade``:

    * Backtrader tracks a Trade per data feed, so each Dec/Apr/Aug roll closes
      one trade and opens another. Counting roll segments as trades inflates
      ``n_trades`` and skews win rate, payoff ratio and profit factor. Roll
      fills carry ``info['is_roll']`` (set by ``FuturesStrategyBase._mark_roll``)
      and move exposure between contracts without opening or closing a row.
    * ``Strategy._notify`` delivers *every* order of a bar before *any* trade of
      that bar, so keying open/close state off ``(feed, tradeid)`` - with
      ``tradeid`` always 0 - let a same-bar reversal or a scale-in overwrite the
      previous entry's direction, price and size.

    Fields: see ``TRADE_LOG_FIELDS``.
    """

    def __init__(self):
        self.trades = []
        self._open = {}       # symbol -> logical trade under construction
        self._closing = []    # flattened rows still collecting their trade pnl
        self._net = {}        # symbol -> net lots across all of its contracts
        self._data_net = {}   # feed -> net lots on that feed
        self._owner = {}      # feed -> deque of rows holding a Trade on it
        self._costs = {}      # symbol -> (multiplier, margin_rate)
        self._next_id = 1

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        size = int(order.executed.size)
        if not size:
            return

        price = float(order.executed.price)
        data = order.data
        symbol = self._symbol_of(order)
        is_roll = bool(self._info(order, 'is_roll', False))

        prev = self._net.get(symbol, 0)
        new = prev + size
        self._net[symbol] = new

        # A row that already went flat never takes another fill: retire it (it
        # keeps collecting pnl from _closing) and let this fill start a new one.
        row = self._open.get(symbol)
        if row is not None and row['flat']:
            del self._open[symbol]
            self._closing.append(row)
            row = None
        if row is None and new != 0:
            # A roll normally cannot start a row, but _maybe_roll_symbol folds a
            # re-issued signal order into its roll order when a product holds
            # both, so accept one here rather than orphan the pnl.
            row = self._open[symbol] = self._begin(symbol, order, new)

        # Feed-level ownership: whoever takes a feed off zero owns the Trade
        # that will eventually close on it. A deque keeps same-bar reversals and
        # "roll plus exit on one bar" from crossing wires.
        dprev = self._data_net.get(data, 0)
        self._data_net[data] = dprev + size
        if dprev == 0 and row is not None:
            self._owner.setdefault(data, collections.deque()).append(row)

        if row is None:
            return

        name = getattr(data, '_name', '') or ''
        if name and name not in row['contracts']:
            row['contracts'].append(name)

        # Commission is taken per fill rather than from Trade.commission so that
        # a position still open at the end of the sample carries the cost of its
        # own entry - Trade.commission is only readable once the trade closes.
        row['commission'] += order.executed.comm

        if is_roll:
            row['roll_days'].add(self._today())
            return

        if abs(new) > abs(prev):
            row['entry_qty'] += abs(size)
            row['entry_notional'] += abs(size) * price
        else:
            row['exit_qty'] += abs(size)
            row['exit_notional'] += abs(size) * price

        mult, margin_rate = self._product_costs(symbol)
        row['size'] = max(row['size'], abs(new))
        row['margin_used'] = max(
            row['margin_used'], abs(new) * price * mult * margin_rate
        )

        if new == 0:
            row['flat'] = True
            row['close_date'] = self._today()

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        queue = self._owner.get(trade.data)
        row = queue.popleft() if queue else None
        if row is None:
            row = self._open.get(self._symbol_of(trade))
        if row is None:
            return
        row['gross_pnl'] += trade.pnl

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def _flush(self):
        """Emit rows whose pnl has fully arrived.

        Safe here because ``LineIterator._next`` runs ``_notify()`` (all order
        then all trade callbacks for the bar) before ``next()``, and analyzers
        are stepped after that.
        """
        while self._closing:
            self.trades.append(self._finalize(self._closing.pop(0)))
        for symbol, row in list(self._open.items()):
            if row['flat']:
                del self._open[symbol]
                self.trades.append(self._finalize(row))

    def prenext(self):
        self._flush()

    def next(self):
        self._flush()

    def stop(self):
        self._flush()
        # Anything still in the market at the end of the sample is booked at the
        # last close, so sum(net_pnl) still reconciles against final equity.
        for symbol, row in list(self._open.items()):
            del self._open[symbol]
            self._settle_open(row)
            self.trades.append(self._finalize(row))

    def _settle_open(self, row):
        strat = self.strategy
        row['open_at_end'] = 1
        row['close_date'] = self._today()
        for name in row['contracts']:
            try:
                data = strat.getdatabyname(name)
            except Exception:
                continue
            pos = strat.getposition(data)
            if not pos.size:
                continue
            comminfo = strat.broker.getcommissioninfo(data)
            mark = float(data.close[0])
            row['gross_pnl'] += comminfo.profitandloss(pos.size, pos.price, mark)
            row['exit_qty'] += abs(pos.size)
            row['exit_notional'] += abs(pos.size) * mark

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _begin(self, symbol, order, net) -> dict:
        return {
            'id': self._take_id(),
            'symbol': symbol,
            'direction': 'long' if net > 0 else 'short',
            'open_date': self._today(),
            'close_date': None,
            'contract': getattr(order.data, '_name', '') or '',
            'contracts': [],
            'roll_days': set(),
            'entry_qty': 0, 'entry_notional': 0.0,
            'exit_qty': 0, 'exit_notional': 0.0,
            'size': 0, 'margin_used': 0.0,
            'gross_pnl': 0.0, 'commission': 0.0,
            'flat': False, 'open_at_end': 0,
        }

    def _take_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _finalize(self, row) -> dict:
        entry = (
            row['entry_notional'] / row['entry_qty'] if row['entry_qty'] else 0.0
        )
        exit_price = (
            row['exit_notional'] / row['exit_qty'] if row['exit_qty'] else entry
        )
        return {
            'trade_id':    row['id'],
            'open_date':   row['open_date'],
            'close_date':  row['close_date'],
            'direction':   row['direction'],
            'symbol':      row['symbol'],
            'contract':    row['contract'],
            'contracts':   '|'.join(row['contracts']),
            'n_rolls':     len(row['roll_days']),
            'open_price':  round(entry, 4),
            'close_price': round(exit_price, 4),
            'size':        row['size'],
            'gross_pnl':   round(row['gross_pnl'], 4),
            'commission':  round(row['commission'], 4),
            'net_pnl':     round(row['gross_pnl'] - row['commission'], 4),
            'margin_used': round(row['margin_used'], 4),
            'open_at_end': row['open_at_end'],
        }

    def _product_costs(self, symbol):
        if symbol in self._costs:
            return self._costs[symbol]
        try:
            costs = product_costs(symbol) if symbol else None
        except KeyError:
            costs = None
        if costs is None:
            pair = (
                getattr(self.strategy.p, 'contract_multiplier', 20),
                getattr(self.strategy.p, 'margin_rate', 0.10),
            )
        else:
            pair = (costs['multiplier'], costs['margin_rate'])
        self._costs[symbol] = pair
        return pair

    def _today(self):
        try:
            return self.strategy.datas[0].datetime.date(0)
        except Exception:
            return None

    @staticmethod
    def _info(order, key, default=None):
        info = getattr(order, 'info', None)
        if info is None:
            return default
        try:
            return info.get(key, default)
        except Exception:
            return getattr(info, key, default)

    def _symbol_of(self, order_or_trade) -> str:
        tagged = self._info(order_or_trade, 'symbol')
        if tagged:
            return normalize_symbol(tagged)
        data = getattr(order_or_trade, 'data', None)
        name = getattr(data, '_name', '') if data is not None else ''
        return parse_product(name) or ''

    def get_analysis(self):
        return self.trades


# ==============================================================================
# Backtest engine main class
# ==============================================================================

class BacktestEngine:
    """
    Wraps Cerebro configuration and execution, and produces structured results.

    Parameters
    ----------
    strategy_class : type
        Strategy class (subclass of FuturesStrategyBase).
    universe : dict
        Output of ``DataManager.get_universe_bundle``.
    config : dict
        Backtest configuration. Keys:
          initial_cash         initial cash (default 100000)
          trade_size           lots per trade (default 1)
          slippage             fill slippage in price points (default 0; buy worse / sell worse)
          strategy_params      extra dict passed to the strategy (optional)
          results_dir          output directory (default backtest/results)
          strategy_name        strategy name (used in file naming)
          execute_on_contracts trade calendar contracts instead of weighted
          symbol               default product when strategy_params omits it
        Margin, commission, and multiplier are read per product from products.py.
    """

    DEFAULT_CONFIG = {
        'initial_cash':         100_000.0,
        'trade_size':           1,
        'slippage':             0.0,
        'strategy_params':      {},
        'results_dir':          os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        ),
        'strategy_name':        'strategy',
        'execute_on_contracts': False,
        'symbol':               None,
    }

    def __init__(self, strategy_class, universe: dict, config: dict = None):
        self.strategy_class = strategy_class
        self.universe = universe
        self.symbols = list(universe['symbols'])
        self.products = universe['products']
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        os.makedirs(self.config['results_dir'], exist_ok=True)
        self.default_symbol = self._resolve_default_symbol()

    def _resolve_default_symbol(self) -> str:
        sp = self.config.get('strategy_params') or {}
        class_default = self.symbols[0]
        try:
            class_default = self.strategy_class.params.symbol
        except Exception:
            pass
        raw = sp.get('symbol', self.config.get('symbol') or class_default)
        symbol = normalize_symbol(raw)
        if symbol not in self.products:
            raise ValueError(
                f"Strategy symbol {symbol!r} is not in loaded products "
                f"{self.symbols}. Set STRATEGY_PARAMS['symbol'] to one of them."
            )
        return symbol

    def _use_contracts(self) -> bool:
        if not self.config.get('execute_on_contracts', False):
            return False
        has_feeds = any(self.products[s].get('contract_feeds') for s in self.symbols)
        if not has_feeds:
            raise ValueError(
                "execute_on_contracts=True but no contract_feeds were provided"
            )
        default_feeds = self.products[self.default_symbol].get('contract_feeds') or {}
        if not default_feeds:
            raise ValueError(
                f"execute_on_contracts=True but {self.default_symbol} has no contract feeds"
            )
        return True

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Run the backtest and return a dict containing:
          cerebro, results, equity_records, trade_logs,
          metrics, log_path
        """
        cerebro = self._build_cerebro()
        print("[BacktestEngine] Starting backtest ...")
        slippage = max(0.0, float(self.config.get('slippage', 0.0) or 0.0))
        if slippage:
            print(f"[BacktestEngine] Slippage: {slippage:g} price points")
        else:
            print("[BacktestEngine] Slippage: off")
        use_contracts = self._use_contracts()
        results = cerebro.run(runonce=False, cheat_on_open=use_contracts)
        strat = results[0]

        equity_records = strat.analyzers.daily_equity.get_analysis()
        trade_logs     = strat.analyzers.trade_log.get_analysis()
        metrics        = self._calc_metrics(equity_records, trade_logs)

        log_path = self._save_trade_log(trade_logs)

        self._print_summary(metrics, log_path)

        return {
            'cerebro':        cerebro,
            'strat':          strat,
            'equity_records': equity_records,
            'trade_logs':     trade_logs,
            'metrics':        metrics,
            'log_path':       log_path,
        }

    # ------------------------------------------------------------------
    # Internal methods: build Cerebro
    # ------------------------------------------------------------------

    def _commission_info(self, symbol: str) -> FuturesCommissionInfo:
        costs = product_costs(symbol)
        per_lot = costs['commission_mode'] == 'per_lot'
        return FuturesCommissionInfo(
            commission=(
                costs['commission_per_lot'] if per_lot else costs['commission_rate']
            ),
            mult=costs['multiplier'],
            margin_rate=costs['margin_rate'],
            per_lot=per_lot,
            margin=1.0,
            commtype=(
                bt.CommissionInfo.COMM_FIXED if per_lot else bt.CommissionInfo.COMM_PERC
            ),
            percabs=not per_lot,
            stocklike=False,
        )

    def _build_cerebro(self) -> bt.Cerebro:
        cfg = self.config
        use_contracts = self._use_contracts()
        default_symbol = self.default_symbol

        cerebro = bt.Cerebro(runonce=False, cheat_on_open=use_contracts)

        default_bundle = self.products[default_symbol]
        cerebro.adddata(
            default_bundle['weighted_feed'],
            name=weighted_feed_name(default_symbol),
        )
        for symbol in self.symbols:
            bundle = self.products[symbol]
            if symbol != default_symbol:
                cerebro.adddata(
                    bundle['weighted_feed'],
                    name=weighted_feed_name(symbol),
                )
            for code, feed in bundle['contract_feeds'].items():
                cerebro.adddata(feed, name=code)

        contract_by_date = {
            symbol: self.products[symbol]['contract_by_date']
            for symbol in self.symbols
        }
        first_print = {
            symbol: self.products[symbol]['first_print']
            for symbol in self.symbols
        }
        default_costs = product_costs(default_symbol)
        default_mult = default_costs['multiplier']

        strat_params = {
            'contract_multiplier':  default_mult,
            'trade_size':           cfg['trade_size'],
            'margin_rate':          default_costs['margin_rate'],
            'execute_on_contracts': use_contracts,
            'symbol':               default_symbol,
            'symbols':              list(self.symbols),
            'first_print':          first_print,
        }
        strat_params.update(cfg.get('strategy_params', {}))
        strat_params['execute_on_contracts'] = use_contracts
        strat_params['contract_by_date'] = contract_by_date
        strat_params['symbol'] = default_symbol
        strat_params['symbols'] = list(self.symbols)
        strat_params['first_print'] = first_print
        cerebro.addstrategy(self.strategy_class, **strat_params)

        cerebro.broker.setcash(cfg['initial_cash'])
        # With shortcash on (backtrader's default) the account is priced through
        # ``getvaluesize(size, close)``, which has no access to the entry price.
        # Turning it off routes pricing through FuturesCommissionInfo.getvalue.
        # For futures both branches compute the same opened/closed values, so
        # this changes how positions are valued, not how cash flows.
        cerebro.broker.set_shortcash(False)

        default_comm = None
        print("[BacktestEngine] Product costs (from products.py):")
        for symbol in self.symbols:
            costs = product_costs(symbol)
            comm_text = (
                f"{costs['commission_per_lot']:g} CNY/lot"
                if costs['commission_mode'] == 'per_lot'
                else f"{costs['commission_rate']:g} of notional"
            )
            print(
                f"  {symbol}: mult={costs['multiplier']}  "
                f"margin={costs['margin_rate']:g}  "
                f"commission={comm_text}"
            )
            comm_info = self._commission_info(symbol)
            cerebro.broker.addcommissioninfo(
                comm_info, name=weighted_feed_name(symbol)
            )
            for code in self.products[symbol]['contract_feeds']:
                cerebro.broker.addcommissioninfo(comm_info, name=code)
            if symbol == default_symbol:
                default_comm = comm_info
        cerebro.broker.addcommissioninfo(
            default_comm or self._commission_info(default_symbol)
        )

        slippage = max(0.0, float(cfg.get('slippage', 0.0) or 0.0))
        cerebro.broker.set_slippage_fixed(
            slippage,
            slip_open=True,
            slip_limit=True,
            slip_match=True,
            slip_out=False,
        )

        if use_contracts:
            cerebro.broker.set_coo(True)

        cerebro.addanalyzer(DailyEquityAnalyzer,  _name='daily_equity')
        cerebro.addanalyzer(TradeLogAnalyzer,     _name='trade_log')
        cerebro.addanalyzer(btanalyzers.SharpeRatio,
                            _name='sharpe',
                            riskfreerate=0.03,
                            annualize=True,
                            timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(btanalyzers.DrawDown,  _name='drawdown')
        cerebro.addanalyzer(btanalyzers.Returns,   _name='returns')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trade_analyzer')

        return cerebro

    # ------------------------------------------------------------------
    # Internal methods: compute metrics
    # ------------------------------------------------------------------

    def _calc_metrics(self, equity_records: list, trade_logs: list) -> dict:
        cfg = self.config
        initial_cash = cfg['initial_cash']

        if not equity_records:
            return {}

        final_equity    = equity_records[-1]['equity']
        total_return    = (final_equity - initial_cash) / initial_cash

        # Daily return series
        daily_returns = [r['daily_return'] for r in equity_records]
        dr_arr = np.array(daily_returns, dtype=float)

        # Sharpe ratio (annualized, risk-free rate 3%)
        risk_free_daily = 0.03 / 252
        excess = dr_arr - risk_free_daily
        sharpe = (
            excess.mean() / excess.std() * math.sqrt(252)
            if excess.std() > 1e-10 else 0.0
        )

        # Max drawdown. Backtrader does not force-liquidate on margin calls, so
        # a leveraged position can carry equity to zero or below; once the
        # running peak itself is non-positive, drawdown-as-a-fraction-of-peak
        # is undefined, so treat that region as a full (100%) drawdown instead
        # of dividing by a zero/negative peak.
        equities = np.array([r['equity'] for r in equity_records], dtype=float)
        running_max = np.maximum.accumulate(equities)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (running_max - equities) / running_max
        drawdowns = np.where(running_max > 0, ratio, 1.0)
        max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        # Maximum drawdown recovery period, measured from the peak before the
        # deepest trough until equity returns to that peak. A missing value
        # means the strategy had not recovered by the end of the sample.
        if max_drawdown <= 1e-12:
            max_drawdown_recovery_days = 0
        else:
            trough_idx = int(np.argmax(drawdowns))
            peak_equity = running_max[trough_idx]
            peak_hits = np.flatnonzero(equities[:trough_idx + 1] == peak_equity)
            peak_idx = int(peak_hits[-1]) if len(peak_hits) else trough_idx
            recovery_indices = np.flatnonzero(
                equities[trough_idx + 1:] >= peak_equity
            ) + trough_idx + 1
            max_drawdown_recovery_days = (
                int(recovery_indices[0] - peak_idx)
                if len(recovery_indices) else None
            )

        # Trade statistics
        n_trades = len(trade_logs)
        if n_trades > 0:
            winning_trades = [t for t in trade_logs if t['net_pnl'] > 0]
            losing_trades  = [t for t in trade_logs if t['net_pnl'] < 0]
            win_rate = len(winning_trades) / n_trades

            avg_win  = (
                sum(t['net_pnl'] for t in winning_trades) / len(winning_trades)
                if winning_trades else 0.0
            )
            avg_loss = (
                abs(sum(t['net_pnl'] for t in losing_trades) / len(losing_trades))
                if losing_trades else 0.0
            )
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else float('inf')
            expectancy = sum(t['net_pnl'] for t in trade_logs) / n_trades
            gross_wins = sum(t['net_pnl'] for t in winning_trades)
            gross_losses = abs(sum(t['net_pnl'] for t in losing_trades))
            profit_factor = (
                gross_wins / gross_losses if gross_losses > 1e-10 else float('inf')
            )
        else:
            win_rate = 0.0
            profit_loss_ratio = 0.0
            expectancy = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0

        # Books check: every yuan the equity curve moved must be accounted for by
        # a logged trade. This is the standing sentinel for the three bugs fixed
        # in this module - it fails if the equity curve picks up a term the trade
        # log does not know about, or if trades are being split or double-counted.
        booked = sum(t['net_pnl'] for t in trade_logs)
        drift = booked - (final_equity - initial_cash)
        if abs(drift) > max(1e-6 * abs(initial_cash), 0.01):
            print(
                f"  [WARN] Trade log does not reconcile with the equity curve: "
                f"sum(net_pnl)={booked:,.2f} vs "
                f"equity change={final_equity - initial_cash:,.2f} "
                f"(drift {drift:,.2f})"
            )

        return {
            'initial_cash':       initial_cash,
            'final_equity':       round(final_equity, 2),
            'total_return':       round(total_return * 100, 4),   # %
            'sharpe_ratio':       round(sharpe, 4),
            'max_drawdown':       round(max_drawdown * 100, 4),   # %
            'max_drawdown_recovery_days': max_drawdown_recovery_days,
            'win_rate':           round(win_rate * 100, 4),       # %
            'profit_loss_ratio':  round(profit_loss_ratio, 4),
            'expectancy':         round(expectancy, 4),
            'profit_factor':      round(profit_factor, 4) if profit_factor != float('inf') else profit_factor,
            'avg_win':            round(avg_win, 4),
            'avg_loss':           round(avg_loss, 4),
            'n_trades':           n_trades,
            'n_winning':          len(winning_trades) if n_trades > 0 else 0,
            'n_losing':           len(losing_trades)  if n_trades > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Internal methods: save trade log
    # ------------------------------------------------------------------

    def _save_trade_log(self, trade_logs: list) -> str:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = self.config['strategy_name']
        filename = f"{name}_trades_{ts}.csv"
        filepath = os.path.join(self.config['results_dir'], filename)

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(trade_logs)

        print(f"[BacktestEngine] Trade log saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Internal methods: print summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(metrics: dict, log_path: str):
        sep = "=" * 50
        print(sep)
        print("  Backtest Result Summary")
        print(sep)
        print(f"  Initial Cash      : {metrics.get('initial_cash', 0):>12,.2f} CNY")
        print(f"  Final Equity      : {metrics.get('final_equity', 0):>12,.2f} CNY")
        print(f"  Total Return      : {metrics.get('total_return', 0):>12.4f} %")
        print(f"  Sharpe Ratio      : {metrics.get('sharpe_ratio', 0):>12.4f}")
        print(f"  Max Drawdown      : {metrics.get('max_drawdown', 0):>12.4f} %")
        recovery_days = metrics.get('max_drawdown_recovery_days')
        recovery_text = (
            f"{recovery_days} trading days"
            if recovery_days is not None else "Not recovered"
        )
        print(f"  MaxDD Recovery    : {recovery_text:>12}")
        print(f"  Win Rate          : {metrics.get('win_rate', 0):>12.4f} %")
        print(f"  Profit/Loss Ratio : {metrics.get('profit_loss_ratio', 0):>12.4f}")
        print(f"  Expectancy        : {metrics.get('expectancy', 0):>12.2f} CNY/trade")
        pf = metrics.get('profit_factor', 0)
        pf_text = f"{pf:.4f}" if pf != float('inf') else "inf"
        print(f"  Profit Factor     : {pf_text:>12}")
        print(f"  Total Trades      : {metrics.get('n_trades', 0):>12}")
        print(f"  Winning Trades    : {metrics.get('n_winning', 0):>12}")
        print(f"  Losing Trades     : {metrics.get('n_losing', 0):>12}")
        print(sep)
        print(f"  Trade Log         : {log_path}")
        print(sep)
