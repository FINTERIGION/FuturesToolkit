# Meta-Labeling — `meta_runner.py`

Meta-labeling splits one decision into two. The **primary model** — any
existing `Strategy` — keeps deciding *direction* and is not modified. A **meta
model**, a binary classifier, judges each entry the primary proposes: *will
this trade make money, net of costs, under this strategy's own exits?* Entries
below a probability threshold are suppressed.

The gate can only subtract. It never opens a trade the primary did not want,
never changes a direction, and **never blocks an exit** — a model that could
suppress a close would be inventing a new exit rule, and the labels it was
trained on (realized P&L under the primary's own exits) would stop describing
the trades it produces.

Works on any strategy `discover_strategies()` finds; nothing in `meta/` is
specific to one.

## Workflow

```bash
# 1. Run the primary unfiltered: how many trades are there to learn from?
python meta_runner.py harvest --strategy double_ma

# 2. Purged walk-forward: does filtering add anything out of sample?
python meta_runner.py walkforward --strategy double_ma \
    --keep-rate 1.0 0.8 0.6 0.5 0.4 --kind rf lr --shuffle-control
# -> results/meta/<Strategy>_<ts>_walkforward.json

# 3. Confirm on the locked window -- once, and only if step 2 said SIGNAL
python meta_runner.py holdout --report results/meta/<...>_walkforward.json \
    --keep-rate 0.5 --kind rf

# 4. Freeze a model trained on everything up to today
python meta_runner.py fit --strategy double_ma --keep-rate 0.65 -o models/dma.joblib

# 5. Today's filtered target positions
python meta_runner.py signal --model models/dma.joblib --update-data
```

A saved model also plugs into the normal runner, which gives the filtered run
the usual charts and trade log:

```bash
python runner.py --strategy double_ma --meta-model models/dma.joblib
```

## Shared flags

`harvest`, `walkforward` and `fit` take the same data flags. Note the defaults
differ from `runner.py`: a meta model needs the longest sample available, and
a filter judged at zero slippage is judged on trades it could not have got.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--strategy` | Required. The primary | — |
| `--symbols` | Products to load | `SA FG CF BU RB HC C JM V` |
| `--start` / `--end` | Sample window | `2015-01-01` / `2026-12-31` |
| `--cash` | Initial equity | `1000000` |
| `--slippage` | Fill slippage in price points | `1.0` |
| `--lots` | Lots per trade | strategy default |
| `--params-from` / `--param` | Primary's params, from an `optimize` report or inline | — |
| `--update-data` | Refresh exchange data first | off |

## `harvest`

Runs the primary unfiltered and reports the sample set: trade count, base rate
(share of winners), and per-symbol breakdown. Below ~300 usable trades there
is nothing to train on, and the later commands say so.

## `walkforward`

| Flag | Meaning | Default |
| --- | --- | --- |
| `--n-folds` | Purged walk-forward folds | `4` |
| `--embargo` | Bars dropped between train and test | `10` |
| `--holdout-frac` | Trailing fraction locked away | `0.20` |
| `--seed` | Model seed | `42` |
| `--keep-rate` | Fractions of entries to keep, one arm each. `1.0` is the no-op self-check | `1.0 0.8 0.65 0.5` |
| `--kind` | Model families: `rf`, `lr` | `rf lr` |
| `--shuffle-control` | Refit on permuted labels to measure the trade-less-often effect | off |
| `--results-dir` | Output directory | `results/meta` |

Writes `<Strategy>_<ts>_walkforward.json` and prints a verdict line. The
verdict is written to be able to say *no*.

## `holdout`

| Flag | Meaning |
| --- | --- |
| `--report` | Required. A `walkforward` `*_walkforward.json` |
| `--keep-rate` | Required. Stated explicitly, so the setting is a decision rather than whatever topped the table |
| `--kind` | `rf` (default) or `lr` |
| `--strategy` | Override the strategy recorded in the report |
| `--force` | Re-use a spent window, or override a `NO EDGE` verdict |

Refuses to run twice on the same report, and refuses a report whose verdict
was not `SIGNAL`: confirming a filter the walk-forward already rejected only
asks a second judge for a different answer, and spends a one-shot window doing
it.

## `fit`

| Flag | Meaning | Default |
| --- | --- | --- |
| `--keep-rate` | Threshold quantile for the frozen model | `0.65` |
| `--kind` | `rf` or `lr` | `rf` |
| `--seed` | Model seed | `42` |
| `-o` / `--out` | Artifact path | `models/meta.joblib` |

Writes `<out>` plus `<out>.json` recording features, params and the training
window. The artifact carries the primary, its params and the training universe,
which is why `signal` and `live_runner.py --model` need nothing else.

## `signal`

Alias for `live_runner.py --model`, kept so `fit` → `signal` reads as one
sequence. See [Live Signals](live.md).

| Flag | Meaning | Default |
| --- | --- | --- |
| `--model` | Required. A `fit` artifact | — |
| `--cash` / `--slippage` | Override what the model was trained with | training values |
| `--symbols`, `--start`, `--end`, `--update-data` | As above | training universe |
| `--results-dir` | Output directory | `results/live` |

## How the backtest stays honest

- **Features come from `open_bar - 1`.** An order placed in `on_bar` fills at
  the *next* bar's open, so the decision bar is the close before the fill.
  Reading at `open_bar` would leak the bar being predicted.
- **Purging.** A trade joins a fold's training set only if it *closed* before
  that fold's training window ended — not merely opened. A trade still running
  at the boundary has a label knowable only later.
- **Embargo.** Inherited from `research/splits.py`, asserted explicitly rather
  than assumed, so a future change to the split scheme fails loudly instead of
  leaking quietly.
- **The baseline runs through the same wrapper**, with a pass-through model.
  Registering features raises the warmup, so an unwrapped baseline would start
  on a different bar and the two curves would not be comparable.
- **The threshold is chosen on training data only**, as the quantile keeping
  `keep_rate` of training entries. `keep_rate` is a round number fixed in
  advance and reported as a curve, not tuned against the curve it is judged
  by. `keep_rate=1.0` must reproduce the baseline exactly — that arm is the
  self-check.

## Reading the result

A filter always removes trades, and removing trades moves Sharpe on its own.
So the equity curve is the *last* thing to look at:

| Diagnostic | What it means |
| --- | --- |
| Out-of-sample AUC | ≈ 0.5 means no signal. Stop; no threshold turns a coin flip into an edge |
| Precision lift | Win rate among kept trades minus win rate over all of them. Positive means the filter *selected* rather than merely subtracted |
| **Shuffled-label control** | The identical pipeline refit on permuted labels. Whatever Sharpe improvement survives that is caused by trading less, not by choosing better |
| Trade-count delta | Keeping half the trades roughly doubles the sampling variance of Sharpe |

**Meta-labeling cannot rescue a primary with no edge.** The classifier learns
which subset of an already-positive-expectancy signal is better; if the full
set is zero or negative, the "best" subset is a slice of noise.
