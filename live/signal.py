"""Today's orders, for any strategy -- with or without a meta-model.

The whole mechanism is one sentence: replay history to the last loaded bar
and read ``engine.pending``. After the final bar's SIGNAL phase there is no
bar N+1 to fill against, so whatever is left queued there is exactly the
order the next session's open would fill.

**Nothing in that reasoning is specific to meta-labeling**, which is why this
lives outside ``meta/``. A plain strategy from ``strategies/`` and a
meta-gated one differ in one place only -- which class the engine was handed
(:meth:`SignalSpec.build_class`) -- and the gated arm merely has a per-symbol
explanation to attach (``last_decisions``) that the plain arm leaves empty.
The two paths share the run, the row building, the report and the JSON, so a
bug can only ever be in both at once.

Two facts about the output, because both are easy to get wrong live:

* ``current_simulated`` is **not** your book. It is what the strategy would
  hold had it run untouched from ``--start`` on this exact history. The
  actionable number is ``target``; reconcile it against what you really hold.
* Sizing that reads ``ctx.equity`` sized off the *simulated* equity, which
  has drifted from your account by the same amount the positions have. For
  such strategies (``lots = 0``) pass your real equity as ``cash``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from core.market import MarketData
from runner import run_single_backtest
from strategies import load_strategy

__all__ = [
    'SignalSpec', 'SignalRow', 'SignalReport', 'compute_signal', 'spec_from_model',
]


@dataclass
class SignalSpec:
    """What to run, and how -- from either source.

    A plain strategy fills this from CLI flags; a meta-gated one fills it
    from a ``meta_runner.py fit`` artifact's sidecar (:func:`spec_from_model`).
    Downstream code never asks which of the two it is holding, it just calls
    :meth:`build_class`.

    ``params`` holds *overrides only* -- what ``runner.resolve_params``
    returns and what ``fit`` stored -- because ``Strategy.__init__`` merges
    the class defaults underneath. Read :attr:`effective_params` when you need
    the values the strategy will actually see (the report does, to decide
    whether it is looking at an equity-sized strategy).
    """

    strategy_cls: type
    params: dict = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    cash: float = 100_000.0
    slippage: float = 0.0
    model: Optional[object] = None
    model_path: Optional[str] = None

    @property
    def strategy_name(self) -> str:
        return self.strategy_cls.__name__

    @property
    def is_meta(self) -> bool:
        return self.model is not None

    @property
    def effective_params(self) -> dict:
        return {**getattr(self.strategy_cls, 'params', {}), **self.params}

    def build_class(self) -> type:
        """The class the engine actually runs.

        ``meta`` is imported here rather than at module scope so that a plain
        signal run never pays for (or requires) sklearn/joblib.
        """
        if self.model is None:
            return self.strategy_cls
        from meta.filter import make_meta_filtered

        return make_meta_filtered(self.strategy_cls, self.model)


def spec_from_model(
    model_path: str,
    *,
    symbols=None,
    cash: Optional[float] = None,
    slippage: Optional[float] = None,
) -> SignalSpec:
    """Rebuild the spec a ``meta_runner.py fit`` artifact was trained under.

    The sidecar is the authority for the primary strategy and its params: a
    signal computed against a *different* primary is a signal from a model
    that never saw it, and would be silently wrong rather than an error.
    Universe, cash and slippage are the three a live run legitimately
    overrides -- cash above all, which should be the real account equity for
    strategies that size off it.
    """
    from meta.model import load_model

    model = load_model(model_path)
    with open(model_path + '.json', encoding='utf-8') as f:
        sidecar = json.load(f)

    return SignalSpec(
        strategy_cls=load_strategy(sidecar['strategy']),
        params=dict(sidecar.get('params') or {}),
        symbols=list(symbols or sidecar['symbols']),
        cash=float(sidecar['cash'] if cash is None else cash),
        slippage=float(sidecar['slippage'] if slippage is None else slippage),
        model=model,
        model_path=model_path,
    )


@dataclass
class SignalRow:
    """One product's line. The meta fields stay ``None`` for a plain run."""

    symbol: str
    contract: str
    current_simulated: int
    target: int
    delta: int
    tradable: bool = True
    stop: Optional[float] = None            # armed, resting: check it tomorrow intrabar
    stop_contract: Optional[str] = None
    stop_rule: Optional[dict] = None        # sticky spec; arms into `stop` after a fill
    primary_side: Optional[int] = None
    proba: Optional[float] = None
    threshold: Optional[float] = None
    verdict: Optional[str] = None
    vetoed_target: Optional[int] = None

    @property
    def action(self) -> str:
        if not self.delta:
            return '-'
        return f'BUY {self.delta}' if self.delta > 0 else f'SELL {-self.delta}'


@dataclass
class SignalReport:
    as_of: str
    spec: SignalSpec
    rows: List[SignalRow]
    simulated_equity: float
    deferred: dict
    engine: object = None

    @property
    def is_meta(self) -> bool:
        return self.spec.is_meta

    @property
    def actionable(self) -> List[SignalRow]:
        return [r for r in self.rows if r.delta]

    def to_dict(self) -> dict:
        spec = self.spec
        out = {
            'as_of': self.as_of,
            'execute_at': 'next session open',
            'strategy': spec.strategy_name,
            'params': spec.params,
            'symbols': list(spec.symbols),
            'cash': spec.cash,
            'slippage': spec.slippage,
            'simulated_equity': self.simulated_equity,
            'signals': [_row_dict(r, meta=self.is_meta) for r in self.rows],
            'deferred': self.deferred,
        }
        if spec.model is not None:
            model = spec.model
            out['model'] = {
                'path': spec.model_path,
                'keep_rate': getattr(model, 'keep_rate', None),
                'threshold': getattr(model, 'threshold_value', None),
                'trained_through': getattr(model, 'trained_through', None),
            }
        return out


_META_FIELDS = ('primary_side', 'proba', 'threshold', 'verdict', 'vetoed_target')


def _row_dict(row: SignalRow, *, meta: bool) -> dict:
    d = {
        'symbol': row.symbol, 'contract': row.contract,
        'current_simulated': row.current_simulated, 'target': row.target,
        'delta': row.delta, 'action': row.action, 'tradable': row.tradable,
        'stop': row.stop, 'stop_contract': row.stop_contract, 'stop_rule': row.stop_rule,
    }
    if meta:
        d.update({name: getattr(row, name) for name in _META_FIELDS})
    return d


def compute_signal(market: MarketData, spec: SignalSpec) -> SignalReport:
    """Replay ``market`` under ``spec`` and read off the next session's order.

    Strategy-agnostic and model-agnostic: the only branch is
    :meth:`SignalSpec.build_class`, and ``last_decisions`` is read with a
    default so a plain strategy -- which has never heard of meta-labeling --
    simply produces rows whose meta fields are empty.
    """
    if not market.n_bars:
        raise ValueError(
            'No bars loaded, so there is no "last bar" to signal from. Widen '
            '--start/--end, or run with --update-data.'
        )

    outcome = run_single_backtest(
        market, spec.build_class(), spec.params, spec.cash, spec.slippage,
    )
    engine = outcome['engine']
    result = outcome['result']
    last = market.n_bars - 1
    as_of = str(market.dates[-1])

    pending = dict(engine.pending)
    decisions = getattr(engine.strategy, 'last_decisions', {}) or {}

    rows = []
    for sym in spec.symbols:
        panel = market.products.get(sym)
        net = int(engine.broker.net_position(sym))
        delta = int(pending.get(sym, 0))
        d = decisions.get(sym) or {}
        # `live_stop` survives the run: a stop armed at the last OPEN and not
        # hit during that bar is precisely the order that should be resting
        # in the real account tomorrow.
        armed = engine.live_stop.get(sym) or {}
        rows.append(SignalRow(
            symbol=sym,
            contract=panel.active_contract(last) if panel else '',
            current_simulated=net,
            target=net + delta,
            delta=delta,
            tradable=bool(panel.can_trade(last)) if panel else False,
            stop=armed.get('price'),
            stop_contract=armed.get('contract'),
            stop_rule=engine.stop_spec.get(sym),
            primary_side=d.get('side'),
            proba=d.get('proba'),
            threshold=d.get('threshold'),
            verdict=d.get('reason'),
            vetoed_target=d.get('target') if d and not d.get('allowed', True) else None,
        ))

    records = result['equity_records']
    equity = float(records[-1]['equity']) if records else float(spec.cash)

    return SignalReport(
        as_of=as_of, spec=spec, rows=rows, simulated_equity=equity,
        deferred=result['deferred'], engine=engine,
    )
