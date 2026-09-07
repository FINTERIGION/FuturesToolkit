# Parameter Optimization — `ft.py optimize`

Optuna search over anchored walk-forward folds, plus overfitting diagnostics and a locked holdout window. Works on any strategy `discover_strategies()` finds; nothing in `research/` is specific to one.

```bash
# 1. See what will be tuned
python ft.py show-space --strategy double_ma

# 2. Search
python ft.py optimize --strategy double_ma --symbols SA CF RB AG --n-trials 200
# -> results/optuna/DoubleMaStrategy_<ts>_best.json

# 3. Evaluate the winner on the window the search never touched
python ft.py holdout --best results/optuna/DoubleMaStrategy_<ts>_best.json
```

The resulting `*_best.json` feeds straight back into a normal run:

```bash
python ft.py backtest --strategy double_ma --params-from results/optuna/<...>_best.json
```

## `show-space`

| Flag | Meaning |
| --- | --- |
| `--strategy` | Required. A discovered short name, or a `module.path:ClassName` reference |

Prints the space that will actually be searched. Params come from the strategy's `space` declaration; anything not declared and not in `fixed_params` gets a heuristic range inferred from its default value.

## `optimize`

Data and window flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--strategy` | Required. A discovered short name, or a `module.path:ClassName` reference | — |
| `--symbols` | Products to load | `SA FG CF C` |
| `--start` / `--end` | Full sample, folds are cut inside it | `2020-01-01` / `2026-12-31` |
| `--cash` | Initial equity | `100000` |
| `--slippage` | Fill slippage in ticks | `0.0` |
| `--update-data` | Refresh exchange data first | off |

Split flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n-folds` | Anchored walk-forward folds | `4` |
| `--embargo` | Bars dropped between train and test | `10` |
| `--holdout-frac` | Trailing fraction locked away from the search | `0.20` |
| `--seed` | TPE sampler seed | `42` |

Search and objective flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n-trials` | Optuna trials | `200` |
| `--probe-samples` | Warmup probe samples (pad must cover the slowest product) | `20` |
| `--lambda-std` | Penalty weight on the fold-to-fold standard deviation of the score | `0.5` |
| `--min-trades-per-year` | How many trades a window is expected to produce per year | `4.0` |
| `--dd-cap` | Drawdown above this is penalized | `0.35` |
| `--sparse-penalty` | How hard to mark down a window that fell short of that expectation | `0.5` |
| `--param name=kind:args` | Override one dimension, e.g. `slow_period=int:20:200`; repeatable | — |
| `--study-name` | Optuna study name | auto |
| `--results-dir` | Output directory | `results/optuna` |

**Objective**: mean out-of-sample Sharpe across folds, adjusted for how few trades the window produced, then penalized for drawdown beyond `--dd-cap`, for forced liquidations, and finally by `--lambda-std ×` the standard deviation across folds. A parameter set that is excellent in one fold and terrible in the next scores worse than a merely decent one.

**Trading less can never raise a score.** Under-trading is handled asymmetrically, on purpose: a *positive* Sharpe is scaled down by the fraction of `--min-trades-per-year` the window actually delivered. On top of that, `--sparse-penalty ×` whatever fraction is missing comes off the score regardless of sign, which puts a floor of `-sparse-penalty` under the do-nothing corner of the space. Raise `--sparse-penalty` toward `1.0` to demand the evidence be there before a configuration counts at all; lower it to let a promising-but-thin one keep being explored.

## `holdout`

| Flag | Meaning |
| --- | --- |
| `--best` | Required. An `optimize` `*_best.json` report |
| `--force` | Re-evaluate a window that has already been spent |

Evaluates the winning params once on the trailing window the search never saw, and records in the report that it has been used.

## Diagnostics

`optimize` reports these alongside the best params:

| Diagnostic | What it answers |
| --- | --- |
| **PBO** (CSCV) | Probability that the in-sample winner is below median out of sample. Above ~0.5 means the selection is noise |
| **Deflated Sharpe** | Sharpe adjusted for the number of trials tried, and for skew and kurtosis of the returns |
| **IS/OOS decay** | How much of the in-sample Sharpe survives out of sample |
| **Plateau check** | Whether the neighbours of the winner also work. A lone spike in the parameter surface is a fit to noise |

**Read them before the equity curve.** A high Sharpe with PBO near 1 and no plateau is a number, not an edge.

PBO and the Deflated Sharpe are both computed on one `[n_trials, n_bars]` matrix of daily returns over the pre-holdout span, and each trial's curve is placed at the bar it actually starts on. A trial can come up short at either end: short at the *head* when its indicators needed more warmup than the pad covers, short at the *tail* when its account blew up and the run stopped there. The bars a trial never traded stand in as zeros, and for a blow-up that reads as a calm stretch rather than a dead account — so when `n_trials_blown_up` in the report is a meaningful share of `n_trials_completed`, treat both diagnostics as optimistic. `optimize` warns when that count is non-zero.
