# Factors — `factor_runner.py`

A factor is a score, not a strategy: a number per product per bar that
claims to predict something. `factors/` judges that claim directly — IC,
quantile buckets, decay, turnover, cross-factor correlation — without
running a backtest, so a bad factor is caught before it spends sizing,
margin, commission, or the strategy-tuning holdout window finding out the
same thing the slow way.

Once a factor looks worth trading, `strategies.factor_bridge` turns it into
a runnable, tunable strategy for free: writing `factors/carry.py` is enough
to get

```bash
python runner.py --strategy factor_carry
python research_runner.py optimize --strategy factor_carry
```

with the factor's own parameters and the trading rule's tuned jointly.

## Workflow

```bash
# 1. What's available?
python factor_runner.py list
python factor_runner.py show-space --factor momentum

# 2. Does it carry information, and how fast does it decay?
python factor_runner.py ic --factor momentum --symbols SA CF AG C JM \
    --start 2016-01-01 --end 2024-12-31 --horizons 1 5 10 20

# 3. Sort the cross-section and see what each bucket actually earned
python factor_runner.py quantiles --factor carry --n-groups 3

# 4. Full single-factor writeup: ic + quantiles + turnover + autocorrelation
python factor_runner.py report --factor momentum --n-groups 3

# 5. Is a new factor saying anything the others don't already?
python factor_runner.py corr --symbols SA CF AG C JM

# 6. Worth trading? Bridge it and tune the trading rule jointly.
python runner.py --strategy factor_carry
python research_runner.py optimize --strategy factor_carry --n-trials 200
```

## Subcommands

| Subcommand | What it answers |
| --- | --- |
| `list` | Every discovered factor, its direction, and its declared params |
| `show-space` | A factor's tunable search space (same resolver `research_runner.py` uses) |
| `ic` | Information coefficient: level, decay across horizons, annual stability |
| `quantiles` | Bucket the cross-section and measure what each bucket earned |
| `report` | `ic` + `quantiles` + turnover + factor autocorrelation, one JSON |
| `corr` | Cross-sectional and IC correlation between every discovered factor |

## Shared flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--factor` | Required (except `corr`). Short name from `list`, or `module:ClassName` | — |
| `--symbols` | Products to load | `SA FG CF C` |
| `--start` / `--end` | Sample window | `2016-01-01` / `2026-12-31` |
| `--horizons` | Forward-return horizons in bars | `1 5 10 20` |
| `--return-source` | `weighted` (OI-weighted continuous, gap-free) or `exec` (the real calendar contract, NaN across a roll) | `weighted` |
| `--n-groups` | Quantile buckets | `3` |
| `--param` | Factor param override: `name=value` | — |
| `--no-plots` | Skip writing PNG charts | off |
| `--results-dir` | Where reports/charts land | `results/factors/` |

## Writing a factor

```python
# factors/carry.py
from core.params import Int
from factors.base import Factor

class CarryFactor(Factor):
    """One-line summary; the class docstring is what `list` prints."""

    params = {'min_oi': 0}
    space = {'min_oi': Int(0, 5000, step=100)}
    direction = 1  # +1: higher score -> higher expected return. -1: the reverse.

    def compute_symbol(self, ctx, sym):
        ...  # -> float64[n_bars]
```

`ctx` is a `factors.base.FactorContext`: `open/high/low/close/settle/volume/oi(sym)`
mirror `strategies.base.SetupContext` exactly, plus `ctx.panel(field)` for a
whole-universe `float64[n_bars, n_symbols]` array and `ctx.contracts(sym)`
for the raw per-contract data a term-structure factor needs (the OI-weighted
series blends every live contract into one price; carry needs to compare
two of them, which only `ctx.contracts` can offer). Override `compute(ctx)`
directly instead of `compute_symbol` when a factor can score the whole
cross-section in one vectorized pass — most of the bundled ones do.

`discover_factors()` picks up any concrete `Factor` subclass anywhere under
`factors/`, private and gitignored modules included, exactly like
`strategies.discover_strategies()`.

## Two conventions that are easy to get backwards

**The validity mask is `tradable`, not `notna`.** A product with no session
today still shows a "price" in the weighted series — `data_manager`
forward-fills it and flattens O/H/L onto the filled close, marking the bar
only with `session == 0`. That bar looks like a perfectly normal quote that
returned exactly 0%; averaging it in silently drags every IC and every
bucket return toward zero. Every function in `research.factor_eval` gates
on `core.market.tradable_mask`, not on whether a number happens to be
non-NaN — a factor implementation never has to think about this itself,
only the evaluation layer does.

**Forward returns are lagged by one bar.** The engine reads a factor's score
in `on_bar` and fills at the *next* open, so a signal on bar `t` cannot
capture bar `t`'s own move. `forward_returns(..., lag=1)` (the default)
lines the measurement up with what the engine could actually have traded.
`lag=0` is the textbook (alphalens) convention and reads higher for any
factor built from same-bar prices — a great-looking report built this way
is a factor being scored against a move it already knew about, not a
finding. `tests/test_factor_eval.py` pins this down in both directions;
read it before trusting a number this module returns.

## Reading a report

- **Cross-sectional IC, not time-series IC.** Every statistic here is
  computed *across products on one bar*, then aggregated over bars — this
  is what a factor used for cross-sectional selection is actually judged on,
  and what makes `--n-groups 3` the right call on a small universe. A factor
  meant for single-product timing needs a different kind of test this tool
  does not run.
- **`|t_stat| < 2` on the aggregate IC is not worth pursuing further.** The
  aggregate is also the number most likely to be misleading on a short,
  regime-heavy sample — see the next point before acting on it either way.
- **Never read the aggregate IC without the annual slice table next to it.**
  A full-sample number can hide a sign flip across years; on this codebase's
  own `momentum_barrier` strategy a full-sample Sharpe of 0.9 hid a swing
  from +2.94 to -1.93 across validation windows. If any year's IC flips sign
  from the rest, check for a structural break (2015 crash, 2016 supply-side
  reform, 2020 pandemic, 2021 dual-control, 2022 Russia/Ukraine) before
  trusting the aggregate at all.
- **Monotonicity matters more than the spread.** A factor whose extreme
  buckets separate but whose middle is scrambled is usually picking up one
  outlier product, not a monotone relationship, and will not survive a
  different universe.
- **High turnover is a cost, not a footnote.** `report`'s turnover line is
  the fraction of the top/bottom bucket replaced per rebalance. This tool
  does not yet net out commission (see "Not built yet" below) — a factor
  with a strong IC and near-total turnover should be treated as unproven
  until it is actually traded (bridge it and look at the real trade log).
- **Correlation has two different questions, and they can disagree.**
  `corr`'s cross-sectional matrix asks whether two factors *rank products
  the same way*; its IC-correlation matrix asks whether they *work at the
  same times*. Two factors can rank identically yet earn in different
  regimes (worth combining), or rank differently yet win and lose together
  (less diversification than the first matrix suggests). Read both.
