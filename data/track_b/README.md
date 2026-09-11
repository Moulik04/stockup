# Track B — private business data

This directory never holds real data (see repo `.gitignore`). It documents the schema the
pipeline expects so the business's sales history can be exported into a matching shape.

## Expected file

One CSV or Excel file, path set via `TRACK_B_DATA_PATH` in `.env`.

## Expected columns

| column | type | notes |
|---|---|---|
| `sku` | string | product identifier, stable over time |
| `date` | date | first day of the period (month or week — see `grain`) |
| `units_sold` | number | sales as a proxy for demand; see the stockout-censoring caveat in `docs/data.md` |
| `unit_price` | number, optional | if price varies over time |
| `on_hand` | number, optional | current stock, only needed for the decision layer, not for backtesting |

## Grain

Monthly by default (`REORDERPOINT_TRACK=b` pipeline config assumes monthly unless told otherwise).
Weekly is also supported if that's what's available — set it in config once real data is seen.

## Known characteristics to expect (from the business, not yet confirmed in data)

- Strong annual seasonality (summer peak for fans/small appliances).
- Fewer, shorter series than Track A (single business vs. thousands of Walmart series).
- Possible zero-inflated or missing months (stockouts, seasonal closures, new SKUs).

This file will be revised once a real export is available and the schema is confirmed.
