# Web Panel

A local browser UI over the same engine the CLI uses: manage the product registry, download data, run backtests and Optuna tuning with live progress and interactive charts, and browse run history. Writing or editing a strategy stays in the editor.

```bash
pip install -e ".[web]"
python ft.py web
```

That serves the pre-built frontend from `web/static/`. To change the frontend, rebuild it:

```bash
cd webui
npm install           # first time only
npm run build         # writes to ../web/static
```

`python ft.py web --host 0.0.0.0` binds beyond localhost; the panel has no authentication and can rewrite the product registry and delete data files, so only do this on a network you trust.

## Pages

| Page | What it does |
| --- | --- |
| Products | Add / edit / delete registered products; per-product candlestick chart with a roll-contract overlay |
| Data | Coverage table (rows, date range, staleness) and a job to download/rebuild per product |
| Backtest | Run any discovered strategy over a date range; equity curve, drawdown, position, price & signals, trade log, per-symbol and per-exit-reason breakdowns |
| Optimize | Optuna anchored walk-forward search with a live trial-by-trial progress chart, fold scores, overfitting diagnostics, and the once-only holdout check |

## Architecture

```
web/                      FastAPI backend
  app.py                    App + SPA static mount (launched by `ft.py web`)
  config.py                 Paths, host/port defaults
  jobs.py                   Background job manager (thread pool + SSE)
  marketcache.py            LRU over research.runner_api.load_market
  store.py                  SQLite run-history index (results/webpanel.db)
  serialize.py              JSON-safe conversion (inf/NaN/date/numpy)
  schemas.py                Pydantic request models
  routers/                  products, data, strategies, backtest, optimize, runs, jobs

webui/                    Vite + React + TypeScript frontend
  src/api/                   Typed fetch client + endpoint functions
  src/charts/                ECharts option builders (candlestick, equity, fold scores, ...)
  src/components/            Card, Table, Drawer, Tabs, EChart, ParamEditor, ...
  src/pages/                 One file per page above
  src/i18n/                  en / zh resource files
```

The backend is a thin layer: every route calls the same functions the CLI calls (`run_single_backtest`, `run_study`, `evaluate_holdout`, `DataManager`, `discover_strategies`).

### Product registry

`datafeed/products.json` is the registry — `datafeed/products.py` loads it, validates writes (`validate_product`/`save_registry`), and keeps every existing helper (`product_costs`, `roll_rule`, `list_products`, ...) unchanged.

Product writes are refused with 409 while any job is running: the engine reads costs and roll rules live, per fill, so an edit landing mid-backtest would silently corrupt that run's numbers. The check and the write happen while no new job can start, so one cannot slip in between them, and work that runs inline rather than on the pool (the holdout evaluation) registers with the job manager for its duration so the guard sees it too.

A data update refuses with 409 if any of its symbols is already being downloaded: the two jobs write the same CSVs from independent fetches, and the loser would overwrite the winner.
