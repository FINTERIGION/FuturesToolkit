# Overfitting Checks — `ft.py validate`

Takes a strategy and the parameters you intend to trade, and answers one question from six directions: **how much of this backtest is the strategy, and how much is this particular slice of history?**

```bash
# 1. See the ranges each parameter will be perturbed over
python ft.py show-space --strategy double_ma

# 2. Check the parameters you actually intend to trade
python ft.py validate --strategy double_ma --symbols SA CF FG C
# -> results/validation/DoubleMaStrategy_<ts>_validation.json

# 3. Replay the same parameters with the full trade log and charts
python ft.py backtest --strategy double_ma --params-from results/validation/<...>_validation.json
```

## The six checks

| Check | What it answers | Flags when |
| --- | --- | --- |
| **Walk-forward consistency** | Does the edge survive the window moving forward? | Out-of-sample folds keep under half the in-sample score |
| **Sub-period stability** | Is the headline number carried by one stretch of history? | The worst period loses money *and* under half the periods are positive |
| **Parameter sensitivity** | Does the result need exactly these numbers? | A one-step move in any dimension costs more than 30% of the score |
| **Block bootstrap** | How much of the Sharpe can the sample size actually pin down? | Under 95% of resamples come out positive |
| **PBO** (CSCV) | Would picking the in-sample winner among the neighbours have held up? | Above 0.5 |
| **Deflated Sharpe** | Does the Sharpe survive correction for how many configurations were tried? | Below 0.95 |

## Flags

Data and window flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--strategy` | Required. A discovered short name, or a `module.path:ClassName` reference | — |
| `--symbols` | Products to load | `SA FG CF C` |
| `--start` / `--end` | Full sample; folds and sub-periods are cut inside it | `2020-01-01` / `2026-12-31` |
| `--cash` | Initial equity | `200000` |
| `--slippage` | Fill slippage in ticks | `0.0` |
| `--update-data` | Refresh exchange data first | off |

Which parameters get checked:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--param NAME=VALUE` | The value to check; repeatable | The strategy's own default |
| `--params-from` | Read `params` from an earlier validate report | — |
| `--lots` | Lots per trade, for strategies that use it | `1` |

Splits and checks:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n-folds` | Anchored walk-forward folds | `4` |
| `--embargo` | Bars dropped between each train window and its valid window | `10` |
| `--holdout-frac` | Trailing fraction excluded from the folds entirely | `0.0` |
| `--n-periods` | Sub-periods to split the sample into; `0` splits by calendar year | `0` |
| `--bootstrap-draws` | Bootstrap resamples | `2000` |
| `--bootstrap-block` | Block length in bars; set it at least as long as a typical holding period | `20` |
| `--trials-tried` | How many parameter sets you tried before settling on this one | grid size |
| `--skip` | Leave a check out: `walkforward`, `subperiod`, `sensitivity`, `bootstrap`, `pbo`, `dsr`; repeatable | — |
| `--fail-on-warn` | Exit non-zero when any check flags, for use in a gate | off |
| `--seed` | Bootstrap seed | `42` |
| `--results-dir` | Output directory | `results/validation` |

## Things worth knowing before you read a report

**`--trials-tried` is the one number the tool cannot observe.** The Deflated Sharpe corrects for how many configurations were tried before one was picked. Left alone it assumes only the neighbourhood grid was ever looked at — a dozen or so — which is almost never true of a strategy someone tuned by hand. Under-declaring it inflates the result. If you tried forty variants before settling, say `--trials-tried 40`.

**`--holdout-frac` defaults to 0.** Holding a window back protects it from *selection*, and nothing here selects: the parameters arrive already chosen. Reserving a tail would only shorten the folds and hide the most recent stretch, which is the one worth seeing.

**IS/OOS decay measures something different here than under a search.** With the parameters fixed, "in sample" is only the earlier and longer stretch. A collapse means the edge did not survive the window moving forward — regime drift, plus whatever fitting happened before this tool saw the parameters — not selection inside this run.

**PBO is computed on a one-step neighbourhood, not an independent search.** It reads as "if I picked among these nearby variants by in-sample performance, would the choice have held up", which is narrower than the same statistic over a real search's trials. Quote it that way.

**A sensitivity verdict of `off-peak` is not a failure.** It means a neighbouring value scored better, so the value in use is not even the local best — the opposite of the lone-spike pattern this check looks for.

**A check can decline to answer.** A window too short to bootstrap, a dimension whose neighbours the strategy's own `constraints` rule out, a return series too short to split into blocks: these report `n/a` and `insufficient`/`not measured` rather than a number. That is deliberate — "did not run" must never read as "ran and found nothing".

## Scale

Every Sharpe in the report is annualized and in excess of `core.metrics.DEFAULT_RISK_FREE_RATE`, on the same scale as `ft.py backtest`'s summary — the bootstrap's point estimate is `compute_metrics`' `sharpe_ratio` exactly, and a test pins it there. The two exceptions are labelled: the Deflated Sharpe's `sr_hat` and `sr0` are **per bar**, an annualization factor away from everything else, because that is the scale the deflation works on.

## `show-space`

| Flag | Meaning |
| --- | --- |
| `--strategy` | Required. A discovered short name, or a `module.path:ClassName` reference |

Prints the range each parameter is perturbed over by the sensitivity check. Ranges come from the strategy's `space` declaration; anything not declared and not in `fixed_params` gets a heuristic range inferred from its default value. Nothing searches these ranges — they set the step size for a one-notch perturbation, and the web panel's parameter editor uses the same bounds to flag a value typed outside them.
