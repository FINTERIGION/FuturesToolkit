# Data Pipeline

Downloads history from CZCE, SHFE and DCE, caches the raw payloads, and builds
two CSVs per product: contract-level bars for execution, and an
open-interest–weighted continuous series for signals.

## `python -m datafeed.data_update`

| Argument | Meaning | Default |
| --- | --- | --- |
| `symbols` | Products to sync, positional | all registered products |
| `--force` | Re-download everything, ignoring the cache | off |
| `--rebuild-only` | Rebuild CSVs from the local cache, no network | off |

`--force` and `--rebuild-only` are mutually exclusive. Without either, the run
is incremental: only trading days missing from the cache are fetched.

```bash
python -m datafeed.data_update              # incremental refresh, all products
python -m datafeed.data_update SA CF        # selected products
python -m datafeed.data_update --force      # full re-download
python -m datafeed.data_update --rebuild-only
```

**First run takes about an hour.** CZCE ships one file per year, but SHFE and
DCE ship one payload per trading day, so a cold cache is thousands of requests.
A failed product is reported and skipped; the rest of the run continues.

## Outputs

| Path | Content |
| --- | --- |
| `cache/{CZCE,SHFE,DCE}/` | Raw exchange payloads, one file per venue-native unit |
| `data/{SYMBOL}.csv` | Contract-level daily OHLC / settle / volume / OI |
| `data/{SYMBOL}_weighted.csv` | OI-weighted continuous series, one row per day |

Both directories are gitignored. The weighted series is what strategies see;
orders fill on the contract-level bars.

## Product registry — `datafeed/products.json`

One entry per product, git-tracked. Adding a product means adding an entry
here — either by hand, or through the [web panel](web.md)'s Products page —
and nothing else in the codebase needs to know about it.
`datafeed/products.py` loads this file at import into the `PRODUCTS` dict and
provides `validate_product`/`save_registry`/`reload_registry` for writers;
every other helper (`product_costs`, `roll_rule`, `list_products`, ...) is
unchanged from before the registry moved out of source code.

| Key | Meaning |
| --- | --- |
| `exchange` | `CZCE` / `SHFE` / `DCE`, selects the download adapter |
| `name`, `name_zh` | Labels used in reports and charts |
| `start_year` | First year to download |
| `multiplier` | Contract size (units per lot) |
| `tick_size` | Minimum price increment; the unit `--slippage` is counted in |
| `margin_rate` | Initial margin as a fraction of notional |
| `commission_rate` | Fee as a fraction of notional |
| `commission_per_lot` | Fee as fixed CNY per lot |
| `main_months` | Delivery months carrying the liquidity, e.g. `(1, 5, 9)` |
| `roll_lead_months` | How many months before delivery to roll out (default `1`) |

`main_months` and `roll_lead_months` are optional; omitting them uses the
module defaults `(1, 5, 9)` and `1`. A contract is rolled out of on the first
calendar day of the month `roll_lead_months` before its delivery month — the
`05` contract is dropped on April 1st. `roll_calendar.py` turns these two keys
into a date → main-contract map.

Contract codes are normalised to uppercase + 4-digit YYMM across all venues
(`RB2610`, `C2601`).
