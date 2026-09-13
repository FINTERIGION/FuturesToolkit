# Writing a Strategy

Subclass `Strategy` from `strategies.base`, precompute indicators in `setup`, trade in `on_bar`. Drop the file anywhere under `strategies/`, then modules are discovered automatically, and the short CLI name is the class name snake-cased without a trailing `_strategy` (`DoubleMaStrategy` → `double_ma`).

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
| `space` | Plausible range per param, `{name: Int / Float / Categorical}` — perturbed one step by the sensitivity check, see [Overfitting Checks](validation.md) |
| `fixed_params` | Params that must never be tuned (default `('lots',)`) |
| `constraints` | Tuple of `callable(params) -> bool`; a perturbed set failing any is skipped by `validate`, e.g. `lambda p: p['fast_period'] < p['slow_period']` |

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

`add_indicator` sets warmup **per product**: it pushes out the bar from which that symbol becomes tradable and leaves the others alone, so a product whose history starts late sits out while the rest of the universe trades.

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
| `ctx.set_take_profit(sym, price=…, distance=…)` / `ctx.cancel_take_profit(sym)` | Target that closes the position when touched intrabar; OCO with the stop |
| `ctx.size_for_risk(sym, stop_distance, risk_pct)` | Lots sized so a stop-out risks `risk_pct` of equity; `0` when the budget will not stretch to one lot |

Orders always fill against that day's calendar contract, so strategy code never names a physical contract or handles a roll itself.

### The protective bracket

`set_stop` and `set_take_profit` are the two legs of one OCO bracket. Both arm at the next OPEN against the fill that just happened, are then checked intrabar against the execution contract's high/low, and close the **whole** position when touched. Whichever fills first cancels the other. Each rule is sticky: it survives a dark session and re-arms after a close and reopen, so cancel it explicitly when a strategy exits for its own reasons.

**The level follows the rule and the position, both.** A resting leg is re-resolved at the next OPEN whenever what it was resolved *from* moves — the position's average cost, its side, or the contract it sits on. So:

- Calling `set_stop` again while one already rests **does** move it, at the next OPEN. That is how a trailing stop is written; no `cancel_stop` first.
- Adding to a position re-anchors a `distance` leg against the new average cost, which is what "`distance` below its cost" means once there is more than one fill in the position. Pass an explicit `price` to pin a level that should not move.
- Reversing re-arms on the correct side of the new position, and a roll moves the level onto the new contract's price scale (see [Backtesting](backtest.md)).
- A dark session changes nothing. It cannot fill either leg, so it does not touch them either.

Pass a `distance`, not a `price`, whenever you can. Signals come off the OI-weighted continuous series but fills land on a real contract, and the two run a basis of a few percent — a level lifted from the weighted series lands in the wrong place, while a distance is anchored on the actual fill and the basis cancels.

A take-profit is a *touch* fill, at the target. That is not the same rule as letting the bar close through the target and leaving at the next open, and the difference is not small. Which rule suits a strategy is an empirical question — check the `exit_reason` split in the run summary rather than assuming.

## Conventions

- Guard every symbol with `can_trade(sym)` before trading it. An order for a product whose own indicators are not valid yet is dropped, and the strategy is never told; the run counts those and warns at the end (`warmup_skips`), because from the outside they look exactly like a signal that never fired.
- Prefer `set_target` over `buy`/`sell`: it is idempotent, so a repeated signal does not stack up a position.
- Cross-sectional strategies read the whole `ctx.symbols` loop as one decision; see `strategies/cross_sectional_momentum.py`.
- Keep `lots` in `params` if you want `--lots` to work.
