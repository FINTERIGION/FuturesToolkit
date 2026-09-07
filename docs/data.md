# Data Pipeline — `python ft.py data`

Downloads history from CZCE, SHFE and DCE, caches the raw payloads, and builds two CSVs per product: contract-level bars for execution, and an open-interest–weighted continuous series for signals.

| Argument | Meaning | Default |
| --- | --- | --- |
| `symbols` | Products to sync, positional | all registered products |
| `--force` | Re-download everything, ignoring the cache | off |
| `--rebuild-only` | Rebuild CSVs from the local cache, no network | off |

```bash
python ft.py data                          # incremental refresh, all products
python ft.py data SA CF                    # selected products
python ft.py data --force                  # full re-download
python ft.py data --rebuild-only
```

**First run takes about an hour.** CZCE ships one file per year, but SHFE and DCE ship one payload per trading day, so a cold cache is thousands of requests.

## Outputs

| Path | Content |
| --- | --- |
| `cache/{CZCE,SHFE,DCE}/` | Raw exchange payloads, one file per venue-native unit |
| `data/{SYMBOL}.csv` | Contract-level daily OHLC / settle / volume / OI |
| `data/{SYMBOL}_weighted.csv` | OI-weighted continuous series, one row per day |

The weighted series is what strategies see; orders fill on the contract-level bars.

## Product registry — `datafeed/products.json`

Adding a product means adding an entry here. `datafeed/products.py` loads this file at import into the `PRODUCTS` dict and provides `validate_product`/`save_registry`/`reload_registry` for writers; every other helper (`product_costs`, `roll_rule`, `list_products`, ...) is unchanged from before the registry moved out of source code.

| Key | Meaning |
| --- | --- |
| `exchange` | `CZCE` / `SHFE` / `DCE`, selects the download adapter |
| `name`, `name_zh` | Labels used in reports and charts |
| `start_year` | First year to download |
| `multiplier` | Contract size (units per lot) |
| `tick_size` | Minimum price increment |
| `margin_rate` | Initial margin as a fraction of notional |
| `commission_rate` | Fee as a fraction of notional |
| `commission_per_lot` | Fee as fixed CNY per lot |
| `main_months` | Delivery months carrying the liquidity |
| `roll_lead_months` | How many months before delivery to roll out |

A contract is rolled out of on the first calendar day of the month `roll_lead_months` before its delivery month — the `05` contract is dropped on April 1st.
