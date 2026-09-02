# Project Layout

```
FuturesToolkit/
├── runner.py                  # Backtest CLI
├── research_runner.py         # Research CLI: show-space / optimize / holdout
├── meta_runner.py             # Meta-labeling CLI: harvest / walkforward / holdout / fit / signal
├── live_runner.py             # Live signal CLI: next session's targets, any strategy
├── plotting.py                # Chart generation
├── core/                      # Engine internals
│   ├── types.py               #   Bar / Order / Fill / OrderType / Reason
│   ├── market.py              #   MarketData / ProductPanel / ContractSeries
│   ├── broker.py              #   cash, positions, margin, commission, forced liquidation
│   ├── ledger.py              #   fill-driven logical trade ledger
│   ├── engine.py              #   four-phase day loop, rolls, stops, deferral
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
│   ├── data_update.py         #   OI-weighted aggregation & CSV output
│   ├── data_manager.py        #   load / align / bundle data for the engine
│   ├── products.py            #   product registry (multiplier, margin, commission, roll months)
│   └── roll_calendar.py       #   date → main-month contract map (from products.py)
├── research/                  # Optuna parameter optimization (strategy-agnostic)
│   ├── space.py               #   search-space resolution (declared / inferred / CLI override)
│   ├── splits.py              #   anchored walk-forward folds + locked holdout window
│   ├── warmup.py              #   exact warmup probing (pad covers the slowest product)
│   ├── runner_api.py          #   single-window backtest with a leak-safe warmup pad
│   ├── objective.py           #   Optuna trial scoring
│   ├── optimize.py            #   study driver + holdout evaluation
│   └── overfit.py             #   PBO (CSCV), Deflated Sharpe, IS/OOS decay, plateau check
├── meta/                      # Meta-labeling (strategy-agnostic)
│   ├── features.py            #   one feature definition, shared by training and inference
│   ├── dataset.py             #   trade log -> (X, y, w), purge rule
│   ├── model.py               #   fitting, thresholds, pass-through / walk-forward / live models
│   ├── filter.py              #   make_meta_filtered(cls, model) -> a gated Strategy
│   └── evaluate.py            #   AUC, precision lift, shuffled-label control
├── live/                      # Live signals (strategy- and model-agnostic)
│   ├── signal.py              #   replay to the last bar, read the order never filled
│   └── report.py              #   terminal table + results/live/*.json
├── tests/                     # pytest suite (synthetic MarketData fixtures)
├── data/                      # Generated CSVs (gitignored)
├── cache/                     # Raw exchange payloads per venue: cache/{CZCE,SHFE,DCE}/ (gitignored)
└── results/                   # Backtest outputs (gitignored), incl. results/optuna, results/meta, results/live
```

## Tests

```bash
pytest
```

Fixtures are synthetic `MarketData`, so the suite runs without any downloaded
data.
