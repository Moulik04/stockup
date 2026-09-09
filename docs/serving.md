# Serving

Phase 6: turns the pipeline into a running service, not just a report generator.

## Production model

`make train` (`reorderpoint/train.py`) fits one LightGBM global model on the **entire** ingested
panel (no held-out fold — Phase 3's backtest already established this model beats the baselines;
this trains the same model on all available history) and persists it via `joblib` to
`models/production/lightgbm.joblib` (gitignored, like all model artifacts). `serve.py` loads this
once at first request (`functools.lru_cache`) rather than retraining per request or per container
start — retraining LightGBM on every request would make the API too slow to be useful.

## Endpoints

- `GET /health` — liveness check.
- `POST /forecast` — `{"series_ids": [...], "horizon": 28}` → P10/P50/P90 for each series/day.
- `POST /reorder` — `{"series_ids": [...], "on_hand": {"series_id": qty, ...}}` → reorder point,
  safety stock, order quantity, stockout probability, and a rationale per series (the same
  `reorderpoint.decision.compute_decisions` policy documented in `docs/decision.md`). Any
  series_id missing from `on_hand` defaults to 0 (Track A has no live inventory feed to draw a
  better default from).
- `GET /metrics` — a small JSON snapshot (series count, panel's last known date, active cost
  config) — not full Prometheus instrumentation, which this project doesn't need yet.

`make score` (`serve.py --batch`) runs the same `/reorder` logic over **every** series in the
panel and writes `outputs/reorder_<date>.csv` — a batch-job path that shares all its logic with
the API path rather than duplicating it.

## The forward-exog simplification

Both `/forecast` and `/reorder` need `future_exog` — the calendar/price/event columns
`LightGBMGlobalModel` expects to know in advance (see `models/lightgbm_global.py`). A real
deployment would source genuine forward-looking values (an actual pricing/promo calendar). This
project doesn't have one, so `future_exog_from_trailing_window` **repeats each series' most
recent `horizon`-day row pattern**, with dates advanced forward — except the columns that are
genuinely computable from the real future date regardless of any external system (`wday`,
`month`, `year`), which are recomputed properly rather than carried over stale.

This is a disclosed proxy, the same kind of simplification as `y`-as-demand-proxy in
`docs/data.md` — not hidden inside the code. It affects price/event/SNAP columns; a Track B
deployment barely notices, since Track B's schema doesn't carry those columns at all
(`features.py`'s feature list already degrades gracefully to whatever columns are present).

## Docker

The image does **not** bundle the raw M5 CSVs or retrain on start — `make train` must be run on
the host (or in your own build stage) first, producing `models/production/lightgbm.joblib`, which
the `Dockerfile` copies in alongside the processed `panel.parquet`. This keeps the image small and
the container's startup fast (the acceptance bar is "answers `/reorder` correctly," not "trains a
model on boot").

```bash
make train                                    # writes models/production/lightgbm.joblib
docker build -t reorderpoint .
docker run -p 8000:8000 reorderpoint
curl -X POST localhost:8000/reorder -H 'content-type: application/json' \
  -d '{"series_ids": ["HOBBIES_1_001_CA_1_validation"]}'
```

## Monitoring

`reorderpoint/monitor.py` is the reusable core for two checks a scheduled job would run once real
outcomes accumulate — no such job exists yet (there's no live traffic to monitor), so this module
is exercised directly by its tests rather than wired into a cron/Actions schedule:

- **Realised-error tracking** (`realised_error`, `rolling_mase`, `flag_error_spikes`): once actual
  sales are known for a date the API already forecast, compare them and track each series'
  rolling MASE, flagging series whose recent error has spiked well above their historical scale.
- **Input drift** (`detect_input_drift`): flags series whose recent sales have shifted more than
  `z_threshold` training-window standard deviations from their historical mean — a simple,
  explainable check (not a full distributional test), consistent with this project's "boring
  tools first" bar. Verified against a synthetically drifted series in `tests/test_monitor.py`.

## Dashboard

`make dashboard` (`reorderpoint/dashboard.py`, Streamlit) — pick a series, see its history, P10/
P50/P90 forecast fan, current reorder recommendation, and the latest backtest/decision report
headlines. Talks to the same `reorderpoint` modules directly (no HTTP calls to the FastAPI
service) — it's a read-only operator view over the same production model and panel, not a client
of the API.
