# Web Panel

A local browser UI over the same engine the CLI uses: manage the product registry, download data, run backtests with live progress and interactive charts, and browse run history. Overfitting checks stay in the CLI (`ft.py validate`). Writing or editing a strategy stays in the editor.

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

Two checks stop a web page you have open from driving the API through your browser. Every request must carry a Host header naming the panel (`FT_WEB_ALLOWED_HOSTS` widens the list), and every write (POST/PUT/PATCH/DELETE) that carries an `Origin` must come from the panel's own address or the Vite dev server, or it is refused with 403. Behind a reverse proxy that rewrites the Host header, list the public origin in `FT_WEB_ALLOWED_ORIGINS`, e.g. `https://panel.example`.

## Pages

| Page | What it does |
| --- | --- |
| Products | Add / edit / delete registered products; per-product candlestick chart with a roll-contract overlay |
| Data | Coverage table (rows, date range, staleness) and a job to download/rebuild per product |
| Backtest | Run any discovered strategy over a date range; equity curve, drawdown, price & signals, trade log, per-symbol and per-exit-reason breakdowns |

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
  routers/                  products, data, strategies, backtest, runs, jobs

webui/                    Vite + React + TypeScript frontend
  src/api/                   Typed fetch client + endpoint functions
  src/charts/                ECharts option builders (candlestick, equity, drawdown, ...)
  src/components/            Card, Table, Drawer, Tabs, EChart, ParamEditor, ...
  src/pages/                 One file per page above
  src/i18n/                  en / zh resource files
```

The backend is a thin layer: every route calls the same functions the CLI calls (`run_single_backtest`, `DataManager`, `discover_strategies`).

### Product registry

`datafeed/products.json` is the registry — `datafeed/products.py` loads it, validates writes (`validate_product`/`save_registry`), and keeps every existing helper (`product_costs`, `roll_rule`, `list_products`, ...) unchanged.

Product writes are refused with 409 while any job is running: the engine reads costs and roll rules live, per fill, so an edit landing mid-backtest would silently corrupt that run's numbers. The check and the write happen while no new job can start, so one cannot slip in between them, and any work that runs inline rather than on the pool registers with the job manager for its duration so the guard sees it too.

A data update refuses with 409 if any of its symbols is already being downloaded: the two jobs write the same CSVs from independent fetches, and the loser would overwrite the winner.
