# Changelog

## Unreleased

### Changed

- **Track B is now a public dataset, UCI Online Retail II**, not a private business's export (that data
  will not be available). A small UK online gift-ware seller with wholesale customers, 2009-2011, cleaned
  to a weekly SKU panel (`make data-b`, `reorderpoint/online_retail.py`, rules and counts in
  `docs/data.md`). Its purpose is external validity: whether the Track A findings replicate at a very
  different real business. The design, the economics (USD at one fixed GBP rate; margin, holding cost and
  lead time as documented assumptions with sensitivity) and the rules for what counts as a replication were
  registered in `docs/track_b.md` before any model was run. **No Track B model results yet.**
- The private-export loader, its `TRACK_B_DATA_PATH` setting and its fixture were removed, and every
  statement that "Track B's real costs" would settle an assumption was reworded: neither track has any.

## v1.2.1 — 2026-09-29

The serving image is now built and queried in CI, and what had been claimed about it is corrected.

### Added

- **CI builds the image and asks the running container real questions.** A fixture panel, model and
  calibration are built, the image is built and run, and `/health`, `/reorder`, `/forecast` and
  `/metrics` are compared to numbers derived by hand (`scripts/docker_smoke.py`; the expectations are
  themselves tested without Docker in `tests/test_docker_smoke.py`).
- The base image's Python minor comes from `.python-version` (a `PYTHON_VERSION` build argument), the
  image uses that Python instead of downloading its own, and a test fails if the Dockerfile, the pin
  and CI drift apart. 79 MB smaller as a side effect (616 MB, measured).
- A subprocess test that the supported import order (lightgbm before hierarchicalforecast) fits
  LightGBM without crashing; the crash and a minimal reproduction are in `DECISIONS.md`.

### Headline claims revised

- **"Docker image" (Phase 6), and its acceptance bar "docker container answers `/reorder`
  correctly"** — never shown to be met. No Dockerfile in this repository's history builds: the
  first failed installing the project (`README.md` not copied), the one after the Python 3.13 bump
  failed building statsforecast (no cp313 wheel, no compiler in the slim image). v1.2.0 fixed the
  build; v1.2.1 is the first version where CI checks it. The `v1.0.0` tag message and `v1.0.1`
  release notes still say "Docker" without qualification.
- **"All direct dependencies ship cp313 wheels" (the 2026-09-11 Python bump)** — false for
  statsforecast 2.0.1, which has wheels for cp39–cp312 only.

## v1.2.0 — 2026-09-29

The served system now shows the uncertainty it acts on, runs a much simpler model, and no longer
loads the model ladder.

### Changed

- **Displayed intervals match the decision.** `/forecast` and the dashboard's fan chart used the
  model's own P10/P90; the reorder point is sized from calibrated pooled residuals, so the band and
  the decision disagreed about uncertainty. The band is now `P50 ± z(0.9) · σ / √lead_time_days`
  from that same calibration (`SafetyStockCalibration.calibrated_interval`), clipped at zero.
  Held out, the reorder point covered realised lead-time demand in 94.8–95.1% of windows against its
  95% target. `/forecast` now returns 503 without the calibration, like `/reorder`, and 409 when the
  calibration was fitted for a different model than the one served.
- **Serving no longer loads the model ladder.** The registry imports each model on first use, so
  importing `reorderpoint.serve` leaves lightgbm, statsforecast and numba unloaded (tested in a fresh
  interpreter). The Docker image installs only a new `serve` dependency group, needs no `libgomp1`,
  and builds: the previous Dockerfile did not (statsforecast has no Python 3.13 wheel).
- **Production model switched to a 28-day moving average** (`TrailingMeanModel`, the ladder's
  MovingAverage rung) instead of LightGBM. The evidence is a tie, not a win: under the shipped
  policy it costs +$11 per fold against LightGBM [−$34, +$51] (paired bootstrap, series resampled;
  the interval is about ±10% of cost, so a difference of that size is not ruled out), and every other
  model tested is tied with LightGBM too. Same forecast as the ladder's MovingAverage (checked row
  for row), fit + predict 0.1 s against 23 s on the 400-series fold, no features, no forward-exog
  proxy, about 1 MB instead of ~110 MB. It gives up a day-to-day shape and any response to price or
  event columns. The artifact moves to `models/production/model.joblib`;
  rerun `make train` (it also refits the calibration, which must follow the model).
- A series the calibration never saw takes the pooled buffer, not a model-quantile fallback the
  served model cannot supply.

### Added

- Divergence report: the model comparison repeated on lead-time totals. It does not support "the
  models agree on lead-time totals" for the clustered models; SeasonalNaive's daily shape does wash
  out over a lead time as long as its season.
- Production-model report: coverage of the calibrated interval against the native one, how often
  the reorder point covers demand, and the served model's equivalence to the ladder's.
- The denylist hook accepts `re:` entries for whole-word patterns.

## v1.1.0 — 2026-09-29

The decision layer was rebuilt and every headline claim was re-tested under it. The main result
changed: which model is cheapest depends on the cost parameters and the ordering policy, and under
the shipped policy the model choice is worth about a tenth of the policy choice. See the README's
"The finding".

### Added

- **Order-up-to policy** `S = s + lot_multiple × lead-time demand` (default `lot_multiple = 2`),
  replacing `S = s`; **calibrated safety stock** in serving, fitted by `make train` / `make calibrate`.
- **Economics with sources:** holding rate stated per year (25%/yr default), lost sales priced at
  gross margin (27.5%), with sensitivity and break-even reports.
- **Safety-stock calibration ablation** (distributional form × pooling), service-level sweep,
  cost-optimal target, and a held-out check of the chosen target.
- **Croston family** (Classic, SBA, TSB) in the model registry (`make croston`).
- **Paired-bootstrap confidence intervals** on model comparisons; **forecast divergence analysis**
  (`make divergence`); **production-model report** (`make production-model`).
- A pre-commit hook that checks staged files against a local, gitignored denylist.

### Headline claims revised

- **"LightGBM beats the naive baseline on simulated cost"** — true only at a holding cost far above
  the plausible range. Now conditional on holding rate, lost-sale cost, service target and ordering
  policy; the ranking flips across plausible values of each.
- **"LightGBM is the cheapest model under the shipped policy"** — it beats SeasonalNaive by $99 per
  fold [$34, $160], but AutoETS is nominally cheaper and every other model tested (cluster members and
  the Croston family) is tied with it.
- **"LightGBM is served for its calibrated intervals"** — its P10/P90 do not feed the reorder
  decision, which is sized from the P50 lead-time mean plus a buffer from pooled empirical residuals.
  They are used only for display (`/forecast`, dashboard fan chart).
- **"Four models are near-interchangeable (>0.95 correlation on every pair)"** — five of six pairs
  clear 0.95 (MovingAverage–LightGBM is 0.934), the agreement is mostly series level rather than
  shape, and cluster members differ by 19–28% of mean demand.
- **"The service gap is a safety-stock sizing problem"** — it was the `S = s` reorder trigger.
- **"A finer safety-stock pooling scheme saves ~9%"** — does not pass its held-out test under the
  repaired policy.
- **"MinTrace reconciliation helps at item level"** — does not survive a bootstrap.

### Changed

- Serving previously sized with the raw quantile-derived method and `S = s`; it now uses the
  calibrated method and the order-up-to policy the results were measured with.
- Python 3.12 → 3.13.

### Not in this release

- NBEATS under the shipped policy is queued for the next Bridges-2 run.
