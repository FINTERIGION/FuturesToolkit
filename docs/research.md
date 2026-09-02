# Parameter Optimization — `research_runner.py`

Optuna search over anchored walk-forward folds, plus overfitting diagnostics
and a locked holdout window. Works on any strategy `discover_strategies()`
finds; nothing in `research/` is specific to one.

```bash
# 1. See what will be tuned
python research_runner.py show-space --strategy double_ma

# 2. Search
python research_runner.py optimize --strategy double_ma \
    --symbols SA FG CF BU RB HC --n-trials 100
# -> results/optuna/DoubleMaStrategy_<ts>_best.json

# 3. Evaluate the winner on the window the search never touched
python research_runner.py holdout --best results/optuna/DoubleMaStrategy_<ts>_best.json
```

The resulting `*_best.json` feeds straight back into a normal run:

```bash
python runner.py --strategy double_ma --params-from results/optuna/<...>_best.json
```

## `show-space`

| Flag | Meaning |
| --- | --- |
| `--strategy` | Required. Strategy to inspect |

Prints the space that will actually be searched. Params come from the
strategy's `space` declaration; anything not declared and not in
`fixed_params` gets a heuristic range inferred from its default value.

## `optimize`

Data and window flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--strategy` | Required | — |
| `--symbols` | Products to load | `SA FG CF BU RB HC C JM V` |
| `--start` / `--end` | Full sample, folds are cut inside it | `2020-01-01` / `2026-12-31` |
| `--cash` | Initial equity | `100000` |
| `--slippage` | Fill slippage in price points | `0.0` |
| `--update-data` | Refresh exchange data first | off |

Split flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n-folds` | Anchored walk-forward folds | `4` |
| `--embargo` | Bars dropped between train and test | `10` |
| `--holdout-frac` | Trailing fraction locked away from the search | `0.20` |
| `--seed` | Sampler seed | `42` |

Search and objective flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n-trials` | Optuna trials | `200` |
| `--probe-samples` | Warmup probe samples (pad must cover the slowest product) | `20` |
| `--lambda-std` | Penalty weight on the fold-to-fold standard deviation of the score | `0.5` |
| `--min-trades-per-year` | Below this, the score is penalized | `4.0` |
| `--dd-cap` | Drawdown above this is penalized | `0.35` |
| `--param name=kind:args` | Override one dimension, e.g. `slow_period=int:20:200`; repeatable | — |
| `--study-name` | Optuna study name | auto |
| `--results-dir` | Output directory | `results/optuna` |

**Objective**: mean out-of-sample Sharpe across folds, penalized for too few
trades, drawdown beyond `--dd-cap`, and forced liquidations, then penalized
again by `--lambda-std ×` the standard deviation across folds. A parameter set
that is excellent in one fold and terrible in the next scores worse than a
merely decent one.

## `holdout`

| Flag | Meaning |
| --- | --- |
| `--best` | Required. An `optimize` `*_best.json` report |
| `--force` | Re-evaluate a window that has already been spent |

Evaluates the winning params once on the trailing window the search never saw,
and records in the report that it has been used. The one-shot rule is the
point: a holdout you can re-query is just another training set.

## Diagnostics

`optimize` reports these alongside the best params:

| Diagnostic | What it answers |
| --- | --- |
| **PBO** (CSCV) | Probability that the in-sample winner is below median out of sample. Above ~0.5 means the selection is noise |
| **Deflated Sharpe** | Sharpe adjusted for the number of trials tried, and for skew and kurtosis of the returns |
| **IS/OOS decay** | How much of the in-sample Sharpe survives out of sample |
| **Plateau check** | Whether the neighbours of the winner also work. A lone spike in the parameter surface is a fit to noise |

Read them before the equity curve. A high Sharpe with PBO near 1 and no
plateau is a number, not an edge.
