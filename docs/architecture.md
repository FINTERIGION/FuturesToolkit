# Project Layout

```
FuturesToolkit/
├── ft.py                      # The CLI: data / backtest / show-space / optimize / holdout / web
├── plotting.py                # Chart generation
├── core/                      # Engine internals
│   ├── types.py               #   Bar / Order / Fill / OrderType / Reason
│   ├── market.py              #   MarketData / ProductPanel / ContractSeries
│   ├── broker.py              #   cash, positions, margin, commission, forced liquidation
│   ├── ledger.py              #   fill-driven logical trade ledger
│   ├── engine.py              #   four-phase day loop, rolls, stops, deferral
│   ├── backtest.py            #   run_single_backtest: one window, engine + metrics
│   ├── metrics.py             #   performance metrics
│   ├── indicators.py          #   TA-Lib NaN guard
│   └── params.py              #   Int / Float / Categorical search-space types
├── strategies/                # Strategy base + examples (tracked) + private modules (gitignored)
│   ├── base.py                #   Strategy / SetupContext / BarContext
│   ├── double_ma.py
│   ├── rsi_mean_reversion.py
│   ├── cross_sectional_momentum.py
│   └── my_strategy.py
├── datafeed/                  # Data pipeline
│   ├── sources.py             #   per-exchange download & cache adapters (CZCE / SHFE / DCE)
│   ├── data_update.py         #   OI-weighted aggregation & CSV output (run_updates)
│   ├── data_manager.py        #   load / align / bundle data for the engine
│   ├── products.py            #   registry loader/validator over products.json (multiplier, margin, commission, roll months)
│   └── roll_calendar.py       #   date → main-month contract map (from products.py)
├── research/                  # Optuna parameter optimization (strategy-agnostic)
│   ├── space.py               #   search-space resolution + CLI param parsing (values and ranges)
│   ├── splits.py              #   anchored walk-forward folds + locked holdout window
│   ├── warmup.py              #   exact warmup probing (pad covers the slowest product)
│   ├── runner_api.py          #   single-window backtest with a leak-safe warmup pad
│   ├── objective.py           #   Optuna trial scoring
│   ├── optimize.py            #   study driver + holdout evaluation
│   └── overfit.py             #   PBO (CSCV), Deflated Sharpe, IS/OOS decay, plateau check
├── web/                       # Web panel backend (FastAPI) -- see docs/web.md
│   ├── app.py                 #   app + SPA static mount (launched by `ft.py web`)
│   ├── config.py              #   paths, defaults, retention caps, path-containment check
│   ├── schemas.py             #   pydantic request models
│   ├── jobs.py                #   background job manager (thread pool + SSE)
│   ├── store.py               #   SQLite run-history index (results/webpanel.db)
│   ├── data_status.py         #   on-disk coverage per product
│   ├── serialize.py           #   JSON-safe conversion (inf/NaN/date/numpy)
│   ├── marketcache.py         #   LRU over research.runner_api.load_market
│   ├── static/                #   built frontend (gitignored; `npm run build` writes here)
│   └── routers/               #   products, data, strategies, backtest, optimize, runs, jobs
├── webui/                     # Web panel frontend (Vite + React + TypeScript)
│   └── src/                   #   api client, ECharts option builders, pages, i18n (en/zh)
├── tests/                     # pytest suite (synthetic MarketData fixtures)
├── data/                      # Generated CSVs (gitignored)
├── cache/                     # Raw exchange payloads per venue: cache/{CZCE,SHFE,DCE}/ (gitignored)
└── results/                   # Backtest outputs (gitignored), incl. results/optuna, results/web
```

## Tests

```bash
pytest
```

Fixtures are synthetic `MarketData`, so the suite runs without any downloaded data.
