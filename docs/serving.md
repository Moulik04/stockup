# Serving

Phase 6: turns the pipeline into a running service, not just a report generator.

## Production model

`make train` (`reorderpoint/train.py`) fits `TrailingMeanModel` on the **entire** ingested panel
(no held-out fold): each series' mean demand over its last 28 days, held flat. It is the ladder's
MovingAverage rung without the statsforecast call — same forecast, checked row for row against the
ladder's implementation in `reports/production_model_*.md` §5 — and it replaced LightGBM in v1.2
because, under the shipped policy, it costs the same to within the noise of the comparison (paired
bootstrap, §2 of that report), needs only each series' own history, and persists about 1 MB
instead of ~110 MB. It is persisted via `joblib` to `models/production/model.joblib` (gitignored,
like all model artifacts). `serve.py` loads it once at first request (`functools.lru_cache`)
rather than refitting per request or per container start.

What the switch gives up: a day-to-day shape (the forecast is flat) and any response to price,
promotion or event columns. The decision consumes a lead-time total, and the evidence says that
total costs the same to get either way on this panel (Track A, HOBBIES, 400 sampled series, four
folds — a tie means the comparison could not separate them, not that they are equal; the 95%
interval on the cost difference is about ±10% of cost). LightGBM stays in the backtest ladder for
exactly this comparison; `models/lightgbm_global.py` is untouched.

## Endpoints

- `GET /health` — liveness check.
- `POST /forecast` — `{"series_ids": [...], "horizon": 28}` → P10/P50/P90 for each series/day.
  P50 is the model's; **P10/P90 come from the safety-stock calibration below, not from the model**
  (the served model has no intervals of its own), so the band and the reorder point are read off the
  same calibrated forecast errors. Like `/reorder`, it returns 503 without the calibration.
- `POST /reorder` — `{"series_ids": [...], "on_hand": {"series_id": qty, ...}}` → reorder point,
  safety stock, order quantity, stockout probability, and a rationale per series (the policy in
  `docs/decision.md`, sized with the **calibrated** safety stock below). Any
  series_id missing from `on_hand` defaults to 0 (Track A has no live inventory feed to draw a
  better default from).
- `GET /metrics` — a small JSON snapshot (series count, panel's last known date, active cost
  config) — not full Prometheus instrumentation, which this project doesn't need yet.

### Safety-stock calibration (`models/production/safety_stock_calibration.joblib`)

Serving used to size safety stock as `z · σ` with σ read off the model's own P10/P90 width — the
method Phase 4 showed loses — while every result in the README was measured with the calibrated
sizing the backtest harness uses (pooled realised forecast residuals, `reorderpoint/safety_stock.py`).
`reorderpoint/calibration.py` closes that gap. `make train` (or `make calibrate`, which leaves the
persisted model alone) holds out the panel's last `lead_time_days`, fits the model on everything
before, forecasts the held-out window **with the same proxy future-exog serving uses**, and persists
the per-series lead-time-demand residuals with the two bucketing variables (zero rate, volume). The
residuals are stored, not a buffer, so the same artifact sizes for any `SERVICE_LEVEL_TARGET`.
`serve.size_decisions` — the single entry point for `/reorder`, `make score` and the dashboard —
applies it through `decision.calibrated_reorder_decision`, the function the harness also calls
(`tests/test_serving_calibration.py` pins the two to the same reorder point for the same series
and forecast).

- **No silent fallback.** If the artifact is missing, `/reorder` returns 503; a calibration fitted
  for a different lead time returns 409. The old raw sizing is not served in either case.
- **One scheme constant.** `safety_stock.DEFAULT_SCHEME` names the scheme both the harness and
  `train.py` use; switching the default is a change to it plus `make calibrate`.
- **Limits.** One calibration window (one residual per series, pooled across ~5.6k series into two
  buckets), so tail estimates rest on that window; and it goes stale as the model does — re-run
  `make calibrate` whenever `make train` is. `GET /metrics` reports the calibration's scheme, date
  and size. On the real model the calibrated buffer averages 2.0× the raw one (6.2 vs 3.1 units;
  higher for 93% of series).

**The interval `/forecast` and the dashboard show** is derived from the same residuals
(`SafetyStockCalibration.calibrated_interval`): `P50 ± z(0.9) · σ / √lead_time_days` per day, with
`σ` the pooled residual std of the series' bucket — the very `σ` the reorder point's buffer is
`z(service level) · σ` of, so daily half-widths add in quadrature to the lead-time uncertainty the
buffer covers. The lower edge is clipped at zero; daily errors are treated as independent (the
same assumption `decision.lead_time_demand_stats` makes in the other direction); it needs the
`normal` form, which is the default scheme. Held out, the reorder point covered realised
lead-time demand in 94.8–95.1% of windows against a 95% target, and the 80% band covered 87% of
lead-time totals and about 91% of days — wider than nominal, the safe side, because demand is
mostly zeros and the lower edge is clipped (`reports/production_model_*.md` §4). A series the
calibration never saw (a SKU launched inside its held-out window) takes the pooled all-series
`σ` and buffer rather than a model-quantile fallback, which the served model could not supply.

`make score` (`serve.py --batch`) runs the same `/reorder` logic over **every** series in the
panel and writes `outputs/reorder_<date>.csv` — a batch-job path that shares all its logic with
the API path rather than duplicating it.

## The forward-exog simplification

Both `/forecast` and `/reorder` pass `future_exog` — the calendar/price/event columns a model like
`LightGBMGlobalModel` expects to know in advance (see `models/lightgbm_global.py`). The served
`TrailingMeanModel` ignores it, so since v1.2 the proxy below only matters if a feature-driven
model is served again (and for the calibration's held-out window, which uses the same path). A real
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
the host (or in your own build stage) first, producing `models/production/model.joblib`, which
the `Dockerfile` copies in alongside the processed `panel.parquet`. This keeps the image small and
the container's startup fast (the acceptance bar is "answers `/reorder` correctly," not "trains a
model on boot").

```bash
make train                                    # writes model.joblib + safety_stock_calibration.joblib
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

`make dashboard` (`reorderpoint/dashboard.py`, Streamlit) — pick a series, see its history, the
forecast with its calibrated 80% band (same source as `/forecast`), current reorder
recommendation, and the latest backtest/decision report headlines. Talks to the same `reorderpoint` modules directly (no HTTP calls to the FastAPI
service) — it's a read-only operator view over the same production model and panel, not a client
of the API.
