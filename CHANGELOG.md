# Changelog

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
