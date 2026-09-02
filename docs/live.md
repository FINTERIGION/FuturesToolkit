# Live Signals — `live_runner.py`

Answers one question: *what should be on at the next session's open?* — for
any strategy `discover_strategies()` finds, with or without a meta-model.

```bash
# A strategy straight out of strategies/
python live_runner.py --strategy double_ma --update-data

# ... replaying a tuned parameter set from `research_runner.py optimize`
python live_runner.py --strategy double_ma \
    --params-from results/optuna/DoubleMaStrategy_<ts>_best.json

# A meta-labeled strategy: the primary, its params and the training universe
# all come out of the `fit` artifact
python live_runner.py --model models/dma.joblib --cash 500000
# -> results/live/signal_<Strategy>_<date>.json
```

## Flags

`--strategy` and `--model` are mutually exclusive and one is required: they
are the two ways of saying *what* to run. Everything after that is shared,
because everything after that is the same computation.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--strategy` | A strategy from `strategies/`, run ungated | — |
| `--model` | A `meta_runner.py fit` artifact: runs the primary it was trained on, entries gated by the model | — |
| `--symbols` | Products to load | training universe with `--model`, else `SA FG CF BU RB HC C JM V` |
| `--start` | Start of the replay. Sets the simulated position and equity the signal is computed from — keep it stable between runs | `2015-01-01` |
| `--end` | End of the replay | `2026-12-31` |
| `--cash` | Set this to your **real** account equity for strategies that size off equity | training cash with `--model`, else `1000000` |
| `--slippage` | Fill slippage in price points | training value with `--model`, else `1.0` |
| `--lots` | `--strategy` only | strategy default |
| `--params-from` / `--param` | `--strategy` only: params from an `optimize` report, or inline | — |
| `--update-data` | Refresh exchange data first. Without it the signal is as of the last bar on disk, which may not be today | off |
| `--results-dir` | Output directory | `results/live` |
| `--verbose` | Debug logging | off |

`meta_runner.py signal --model ...` is an alias for the `--model` form.

## How it works

Replay history to the last loaded bar and read `engine.pending`. After the
final bar's signal phase there is no bar N+1 to fill against, so what is left
queued there is exactly the next session's order.

Nothing about that is specific to meta-labeling — a gated strategy is just a
different class handed to the same engine — which is why `live/` sits outside
`meta/`, and why both paths share the run, the rows, the table and the JSON.

## Output

```
  sym   contract    sim pos  target    action   side    proba   thresh       stop  note
  SA    SA601             0      +1     BUY 1   long   0.7143   0.5821          -  allowed
  RB    RB2601           -2      -2         -  short        -        -    3512.00  hold
  CF    CF601             0       0         -   long   0.4210   0.5821          -  VETOED (primary wanted +3)
```

| Column | Meaning |
| --- | --- |
| `sym` / `contract` | Product, and the main contract as of the last bar |
| `sim pos` | Position simulated from `--start` |
| `target` | Position that should be on after the next open — **the actionable column** |
| `action` | The order that gets you from `sim pos` to `target` |
| `side` / `proba` / `thresh` | Model gate; present only when a model is gating |
| `stop` | Protective stop armed at the last open and never hit — the order that should be resting in your account tomorrow |
| `note` | `allowed` / `hold` / `VETOED (primary wanted …)` |

The same rows are written to `results/live/signal_<Strategy>_<date>.json`.

## Caveats

The command prints these for itself:

1. `sim pos` is simulated from `--start` and **will** drift from a real
   account. Reconcile `target` against your real book.
2. For strategies that size off equity (`lots = 0`), the lot counts were sized
   on the **simulated** equity. Pass `--cash` equal to your actual account
   equity, or they are wrong.
3. `contract` is the main contract as of the last bar. Re-check it if a roll
   falls on the next session.
