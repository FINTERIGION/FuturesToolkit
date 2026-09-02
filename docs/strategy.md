# Writing a Strategy

Subclass `Strategy` from `strategies.base`, precompute indicators in `setup`,
trade in `on_bar`. Drop the file anywhere under `strategies/` — modules are
discovered automatically, so no registration is needed, and the short CLI name
is the class name snake-cased without a trailing `_strategy`
(`DoubleMaStrategy` → `double_ma`).

```python
import talib
from .base import Strategy

class MyStrategy(Strategy):
    params = {'period': 20, 'lots': 1}

    def setup(self, ctx):
        for sym in ctx.symbols:
            ctx.add_indicator('sma', sym, talib.SMA(ctx.close(sym), self.p['period']))

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            if not ctx.can_trade(sym):
                continue
            pos = ctx.position(sym)
            close = ctx.bar(sym).close
            if pos == 0 and close > ctx.ind('sma', sym):
                ctx.set_target(sym, self.p['lots'])
            elif pos > 0 and close < ctx.ind('sma', sym):
                ctx.close(sym)
```

Start from `strategies/my_strategy.py` if you want a template.

## Class attributes

| Attribute | Meaning |
| --- | --- |
| `params` | Dict of defaults; instance values live on `self.p`, overridden via `MyStrategy(**kw)`, `--param` or `--params-from` |
| `space` | Tunable search space, `{name: Int / Float / Categorical}` — see [Parameter Optimization](research.md) |
| `fixed_params` | Params that must never be tuned (default `('lots',)`) |
| `constraints` | Tuple of `callable(params) -> bool`; a trial failing any is pruned, e.g. `lambda p: p['fast_period'] < p['slow_period']` |

## Lifecycle

| Method | When |
| --- | --- |
| `setup(ctx)` | Once, before the day loop. Register full-series indicators here |
| `on_bar(ctx)` | Once per trading day, after warmup. Orders placed here fill at the **next** open |
| `on_finish(engine)` | Optional, once after the day loop |

## `SetupContext`

Full-series arrays of the OI-weighted series, NaN-guarded:

| Accessor | Returns |
| --- | --- |
| `ctx.symbols` | Products in this run |
| `ctx.open/high/low/close/settle/volume/oi(sym)` | `np.ndarray` over the whole loaded history |
| `ctx.add_indicator(name, sym, array)` | Registers a precomputed indicator |

`add_indicator` sets warmup **per product**: it pushes out the bar from which
that symbol becomes tradable and leaves the others alone, so a product whose
history starts late sits out while the rest of the universe trades.

## `BarContext`

Handed to `on_bar`, after INTRABAR and before SETTLE:

| Accessor | Meaning |
| --- | --- |
| `ctx.bar(sym)` | `Bar(open, high, low, close, settle, volume, oi)` on the weighted series |
| `ctx.ind(name, sym)` | Registered indicator value at this bar |
| `ctx.position(sym)` | Net lots, signed |
| `ctx.equity` / `cash` / `margin_used` / `available` | Account state |
| `ctx.can_trade(sym)` | Own warmup done + listed + real print today + mapped contract live today |
| `ctx.contract(sym)` | Today's calendar contract code, e.g. `'SA509'` |
| `ctx.queued_orders()` | Orders queued this bar, not yet filled |
| `ctx.set_target(sym, lots)` | Target-position order; flips side in one order |
| `ctx.buy/sell(sym, lots)` / `ctx.close(sym)` | Delta orders / flatten |
| `ctx.set_stop(sym, price=…, distance=…)` / `ctx.cancel_stop(sym)` | Protective stop, armed at the next open |
| `ctx.size_for_risk(sym, stop_distance, risk_pct)` | Lots sized so a stop-out risks `risk_pct` of equity; `0` when the budget will not stretch to one lot (traced at DEBUG — run `--verbose` if a strategy trades less than expected) |

Orders always fill against that day's calendar contract, so strategy code
never names a physical contract or handles a roll itself.

## Conventions

- Guard every symbol with `can_trade(sym)` before trading it.
- Prefer `set_target` over `buy`/`sell`: it is idempotent, so a repeated
  signal does not stack up a position.
- Cross-sectional strategies read the whole `ctx.symbols` loop as one
  decision; see `strategies/cross_sectional_momentum.py`.
- Keep `lots` in `params` if you want `--lots` to work.
