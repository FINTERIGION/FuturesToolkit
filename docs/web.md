# Web Panel

A local browser UI over the same engine the CLIs use: manage the product
registry, download data, run backtests and Optuna tuning with live progress
and interactive charts, read live signals, and browse run history. English by
default, with a 中文 toggle. Writing or editing a strategy stays in the editor
— the panel runs what is already in `strategies/`.

```bash
pip install -e ".[web]"          # fastapi + uvicorn
python -m web.app                # http://127.0.0.1:8000
```

That serves the pre-built frontend from `web/static/`. To change the
frontend, rebuild it:

```bash
cd webui
npm install           # first time only
npm run build         # writes to ../web/static
```

For frontend development with hot reload, run the two halves separately —
`npm run dev` starts Vite on `:5173` and proxies `/api` to the backend on
`:8000`:

```bash
python -m web.app --port 8000    # terminal 1
cd webui && npm run dev          # terminal 2 -> http://localhost:5173
```

`python -m web.app --host 0.0.0.0` binds beyond localhost; the panel has no
authentication and can rewrite the product registry and delete data files, so
only do this on a network you trust.

Two request fields name a filesystem path, and both are confined on purpose,
because over HTTP they come from whatever can reach the port rather than from
the person at the keyboard:

- The static handler serves only files that resolve inside `web/static/`.
- A meta-model path (the Signals page's model box, and `meta_model` on a
  backtest) must resolve inside `models/` — `models/x.joblib`, a bare
  `x.joblib`, or an absolute path under it all work, anything else is a 422.
  `joblib.load` unpickles, which runs whatever the file says to run, so the
  panel will not load one from an arbitrary path the way the CLI's `--model`
  will.

## Pages

| Page | What it does |
| --- | --- |
| Products | Add / edit / delete registered products; per-product candlestick chart with a roll-contract overlay |
| Data | Coverage table (rows, date range, staleness) and a job to download/rebuild per product |
| Backtest | Run any discovered strategy over a date range; equity curve, drawdown, position, price & signals, trade log, per-symbol and per-exit-reason breakdowns |
| Optimize | Optuna anchored walk-forward search with a live trial-by-trial progress chart, fold scores, overfitting diagnostics, and the once-only holdout check |
| Factors | Judge a predictive score directly: IC decay, annual stability, quantile-bucket net value, turnover, factor autocorrelation, and a cross-factor correlation heatmap -- see [Factors](factors.md) |
| Signals | Next session's target positions, for a plain strategy or a meta-gated one |
| Runs | Every backtest/optimize run, indexed for reopening and side-by-side comparison |

## Architecture

```
web/                      FastAPI backend
  app.py                    App + SPA static mount + `python -m web.app`
  config.py                 Paths, host/port defaults
  jobs.py                   Background job manager (thread pool + SSE)
  marketcache.py            LRU over research.runner_api.load_market
  store.py                  SQLite run-history index (results/webpanel.db)
  serialize.py              JSON-safe conversion (inf/NaN/date/numpy)
  schemas.py                Pydantic request models
  routers/                  products, data, strategies, backtest, optimize, factors, signals, runs, jobs

webui/                    Vite + React + TypeScript frontend
  src/api/                   Typed fetch client + endpoint functions
  src/charts/                ECharts option builders (candlestick, equity, fold scores, ...)
  src/components/            Card, Table, Drawer, Tabs, EChart, ParamEditor, ...
  src/pages/                 One file per page above
  src/i18n/                  en / zh resource files
```

The backend is a thin layer: every route calls the same functions the CLIs
call (`run_single_backtest`, `run_study`, `compute_signal`,
`DataManager`, `discover_strategies`) — no parallel engine, no duplicated
logic.

### Jobs

A backtest returns in seconds; an Optuna study can run for hours; a
multi-symbol data download is close to an hour on a cold cache. All three run
on a small thread pool (`web/jobs.py`) rather than blocking the request.
`GET /api/jobs/{id}/stream` is a Server-Sent-Events feed;
`POST /api/jobs/{id}/cancel` is best-effort — an Optuna study stops cleanly
after its current trial, a data update stops before its next symbol, a single
backtest cannot be interrupted mid-run.

The stream carries three frame types, all `data:` lines holding one JSON
object with a `type`:

| `type` | Payload | Notes |
| --- | --- | --- |
| `log` | `level`, `message` | The engine's own `logging` output, forwarded live |
| `progress_data` | `entries` | New telemetry only (one entry per Optuna trial), not the list so far |
| `state` | the job record | Everything except `progress_data` and `result` |

Deltas rather than snapshots because a study appends one trial entry per
callback and the stream emits a frame per callback: re-sending the list each
time made the traffic quadratic in trial count. The frontend's `useJob` hook
accumulates them, so page code still reads a complete `state.progress_data`.

Per-job log is a ring buffer (`JOB_LOG_BUFFER`, 2000 lines). A stream that
cannot keep up, or one attaching to a job that already overflowed it, is told
how many lines it missed rather than being handed a log that looks whole.

`result` is deliberately absent from `state` frames — a backtest or optimize
report can be arbitrarily large — so fetch it from `GET /api/jobs/{id}` once
the job is terminal. If the stream drops (a restarted backend, a sleeping
laptop), the UI falls back to polling that same endpoint and says so in the
job header; status, progress and the final result still arrive, live log
lines do not.

### Product registry

`datafeed/products.json` is the registry now — `datafeed/products.py` loads
it, validates writes (`validate_product`/`save_registry`), and keeps every
existing helper (`product_costs`, `roll_rule`, `list_products`, ...)
unchanged. **Product writes are refused with 409 while any job is running**:
the engine reads costs and roll rules live, per fill, so an edit landing
mid-backtest would silently corrupt that run's numbers. The check and the
write happen while no new job can start, so one cannot slip in between them,
and work that runs inline rather than on the pool (the holdout evaluation)
registers with the job manager for its duration so the guard sees it too.

`code` is a path segment on every verb — `POST/PUT/DELETE /api/products/{code}`.

A data update refuses with 409 if any of its symbols is already being
downloaded: the two jobs write the same CSVs from independent fetches, and
the loser would overwrite the winner.

### Run history vs. Optuna reports

Two things track "past runs," on purpose:

- `results/optuna/*_best.json` — Optuna's own report per study, written by
  `research.optimize.run_study` exactly as the CLI writes it. This is the
  authority for a study's search space, fold metrics, and diagnostics.
- `results/webpanel.db` (`web/store.py`) — a small index the panel adds on
  top, one row per backtest/optimize run, so the Runs page can list and
  compare across kinds without re-parsing every JSON report. Equity curves
  and trade logs live in `results/web/{run_id}.json`, not in the database
  row, to keep the index small.

Factor reports follow the Optuna pattern, not the Runs-page one: they are
not "runs" (no equity curve, no strategy) and never touch `webpanel.db`.
`research.factor_report.save_report` writes `results/factors/*.json`
identically whether called from `factor_runner.py` or from
`web/routers/factors.py`, and `GET /api/factors/reports` lists that
directory directly -- a report saved from either side shows up for both.

### JSON safety

`compute_metrics` can return literal `float('inf')` (a strategy with no
losing trades has an undefined profit factor). Python's `json` module emits
a bare `Infinity` token for that, which is not valid JSON and throws in the
browser. `web/serialize.py`'s `jsonable()`/`annotate_inf_metrics()` convert
every response — `inf`/`-inf`/`NaN` become `null` plus a sibling
`<field>_is_inf` flag, so the UI renders `∞` rather than a blank.

## Tests

```bash
pytest tests/test_registry.py tests/test_web_api.py -q
```

`test_registry.py` covers the JSON registry's load/validate/save/reload
round trip. `test_web_api.py` runs the FastAPI app in-process
(`TestClient`) against the real engine and real data on disk — including one
full backtest and one full (tiny) optimize run through the HTTP surface, and
an explicit check that a metrics payload containing `inf` survives
`json.dumps`/`json.loads` as strict JSON.
