# Stockup

Forecast demand with calibrated uncertainty, then turn the forecast into a **reorder decision**
(how much, when) that minimises stockout plus holding cost — proven against naive baselines in
honest, leakage-free backtests. The output is a reorder table, not a chart.

> Status: Phase 7 (deep model, stretch) done. LightGBM wins on both accuracy and simulated dollars
> (Phase 4) and remains the production model — a V100-trained NBEATS beats it on raw accuracy but
> loses badly on decision cost (Phase 7), a sharper repeat of the same lesson Phase 4 taught.
> MinTrace reconciliation helps modestly at the most granular hierarchy levels (Phase 5); a
> FastAPI service, batch scoring, Docker image, CI, monitoring, and a Streamlit dashboard sit on
> top of it (Phase 6) — see "Results" and [`docs/serving.md`](docs/serving.md).

## Why

Most forecasting portfolio projects stop at RMSE on a holdout set. This one goes to the decision
that actually costs money: given a forecast, what do you order, and does that choice beat the
naive policy on simulated dollars, not just accuracy.

## Data

- **Track A (public, in this repo):** [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) — Walmart daily unit sales, hierarchical (item × store × dept × category × state).
- **Track B (private):** a real small-business SKU-level sales history, run through the identical
  pipeline via a config switch. Never committed; results reported in aggregate only.

See [`docs/data.md`](docs/data.md).

## Quickstart

**macOS prerequisite:** `brew install libomp` (LightGBM's OpenMP runtime).

```bash
make setup      # uv sync + pre-commit install
make data       # download and ingest Track A (M5), HOBBIES only — fast day-to-day iteration
make backtest   # rolling-origin backtest, writes reports/backtest_<date>.md
make decide     # reorder policy + cost simulation, writes reports/decision_<date>.md
make data-full  # full M5, all 3 categories — needed for reconcile
make reconcile  # hierarchical reconciliation experiment, writes reports/reconciliation_<date>.md
make train      # fit + persist the production LightGBM model (models/production/lightgbm.joblib)
make serve      # FastAPI at :8000 — GET /health, POST /forecast, POST /reorder, GET /metrics
make score      # batch-score every series, writes outputs/reorder_<date>.csv
make dashboard  # Streamlit operator view — history, forecast fan, reorder rec, monitoring
make test       # unit + leakage tests (includes a real end-to-end smoke backtest)
```

Docker: `make train && docker build -t stockup . && docker run -p 8000:8000 stockup` — see
[`docs/serving.md`](docs/serving.md) for the full walkthrough (including a `curl` example).

## Architecture

```
raw data → ingest → canonical long table → feature pipeline (point-in-time safe)
    → model registry (naive → statistical → global LightGBM → [deep])
    → rolling-origin backtester → quantile forecasts (P10/P50/P90)
    → decision layer (reorder point, safety stock, order qty, cost sim)
    → FastAPI + Streamlit, with drift/accuracy monitoring
```

## Results

**Accuracy (Phases 2–3)** — Track A, HOBBIES category, 400-series subsample, 4 rolling-origin
folds, 28-day horizon. Full table and methodology in
[`reports/backtest_2026-09-03.md`](reports/backtest_2026-09-03.md).

| model | MASE | RMSSE | coverage (nominal 80%) |
|---|---|---|---|
| AutoTheta | 1.529 | 0.823 | 61.6% |
| MovingAverage | 1.530 | 0.820 | 62.2% |
| AutoETS | 1.532 | 0.814 | 63.3% |
| **LightGBM** | **1.587** | 0.823 | **81.2%** |
| SeasonalNaive | 1.633 | 1.088 | 73.2% |

LightGBM beats seasonal naive on MASE in 4/4 folds and is the only model that clears the ±5-point
quantile-coverage bar (81.2% vs. 80% nominal) — the statistical baselines' P10/P90 bands are all
12–22 points too narrow (see the report's "Quantile calibration" section).

**Decision-layer cost simulation (Phase 4)** — same folds/models, run through the newsvendor
policy in [`docs/decision.md`](docs/decision.md) and a lost-sales (s, S) inventory simulation.
Full table in [`reports/decision_2026-09-05.md`](reports/decision_2026-09-05.md).

| model | mean total cost / fold | holding cost | stockout cost | fill rate |
|---|---|---|---|---|
| **LightGBM** | **$14,543** | $7,225 | $7,318 | 81.5% |
| AutoETS | $14,730 | $7,097 | $7,633 | 80.7% |
| AutoTheta | $14,831 | $7,383 | $7,448 | 81.1% |
| MovingAverage | $14,914 | $7,291 | $7,624 | 80.7% |
| SeasonalNaive | $15,224 | $8,582 | $6,642 | 83.2% |

LightGBM now wins on cost too — $681/fold (4.5%) cheaper than SeasonalNaive — but that wasn't true
on the first pass, and the reason why is worth keeping visible rather than quietly fixing and
moving on:

**What went wrong first, and why.** The initial version sized each model's safety stock straight
from its own stated P10/P90 width. That first run had LightGBM *losing* on cost despite winning on
accuracy: its interval was ~18% numerically narrower than SeasonalNaive's, which is enough to
under-provision safety stock even with *better* marginal quantile coverage (81.2% vs. 73.2%) —
because interval width isn't a normalized, cross-model-comparable "uncertainty" unit, it's just
whatever each model's training objective happens to produce. A model didn't have to be wrong to
lose; it only had to be narrower.

**The fix:** size safety stock from each model's own *realised* forecast error (`actual -
predicted`), pooled across ~400 series into two intermittency buckets rather than trusting any
model's self-reported interval — full design and the rejected alternatives in
[`docs/decision.md`](docs/decision.md). Every model's simulated cost went *up* under the fix (the
old method was under-provisioning safety stock across the board, not just for LightGBM), but the
ranking now matches accuracy — the accuracy winner should also win on decision cost, which is the
whole point of judging models by decision effect rather than RMSE alone.

**Hierarchical reconciliation (Phase 5)** — full M5 ingested (30,490 series, all 3 categories,
kept off the memory-safe path via parquet predicate pushdown), scoped to a 6,098-series hierarchy
(2 stores, one per state, full category/department tree under each), 400-series bottom-level
subsample. Bottom-up and MinTrace via `hierarchicalforecast`, AutoETS as the base model
(LightGBM's price/calendar features don't have a sensible definition for a synthetic node like
"all of California"). Full table in
[`reports/reconciliation_2026-09-07.md`](reports/reconciliation_2026-09-07.md).

| level | base MASE | BottomUp | MinTrace |
|---|---|---|---|
| state | 0.747 | 0.774 (+3.6%) | 0.755 (+1.1%) |
| category | 0.794 | 0.804 (+1.3%) | 0.794 (+0.1%) |
| department | 0.849 | 0.856 (+0.9%) | 0.848 (**-0.1%**) |
| item (bottom) | 1.329 | 1.329 (0.0%) | 1.327 (**-0.1%**) |

**Naive BottomUp reconciliation is not recommended here** — it underperforms the unreconciled
base forecast at every aggregate level, because it's just a sum of noisy bottom-level (item ×
store) forecasts, and those are much noisier than a directly-fit aggregate (MASE 1.33 vs. 0.75).
**MinTrace helps modestly at the two most granular levels and is roughly neutral elsewhere** —
consistent with, not contrary to, reconciliation theory: its covariance-weighted blend is
specifically designed not to over-trust the noisiest level the way BottomUp does. A real effect,
but a modest one (sub-2%), not a dramatic before/after.

**Serving (Phase 6)** — the pipeline is now a running service, not just a report generator:
`reorderpoint/train.py` persists the production LightGBM model; `serve.py` exposes it as a FastAPI
app (`/health`, `/forecast`, `/reorder`, `/metrics`) and a `make score` batch job sharing the same
decision logic; a Dockerfile packages it (image doesn't train on boot — see docs/serving.md);
`.github/workflows/ci.yml` runs lint + the full test suite (including a real, not mocked,
end-to-end smoke backtest) on every push; `monitor.py` provides rolling-MASE error-spike detection
and input-drift detection (verified against a synthetically drifted series in
`tests/test_monitor.py`); and a Streamlit dashboard (`make dashboard`) gives an operator a
per-series view: history, forecast fan, reorder recommendation, latest backtest/decision report
headlines, and a live drift check against the panel's own history. Forward-looking exog data
(future price/events, which no real system feeds this project) is a disclosed proxy documented in
`docs/serving.md`, not a hidden assumption.

**Deep model on Bridges-2 (Phase 7, stretch)** — an N-BEATS model (`neuralforecast`), trained on
a PSC Bridges-2 V100 GPU in fp16 (V100/Volta doesn't support bf16 tensor cores or FlashAttention
properly, so both are avoided entirely — verified against the real hardware before writing the
integration, see [`docs/bridges2.md`](docs/bridges2.md)), same 400-series/4-fold setup as every
earlier phase for a direct comparison. Full tables in
[`reports/backtest_deep_2026-09-10.md`](reports/backtest_deep_2026-09-10.md) and
[`reports/decision_deep_2026-09-10.md`](reports/decision_deep_2026-09-10.md).

| model | MASE | coverage (nominal 80%) | mean total cost / fold | fill rate |
|---|---|---|---|---|
| **NBEATS** | **1.108** | **84.5%** | $17,209 | 71.5% |
| LightGBM | 1.587 | 81.2% | **$14,543** | 81.5% |
| SeasonalNaive | 1.633 | 73.2% | $15,224 | 83.2% |

NBEATS is the most accurate model in the whole project by a wide margin — better MASE than
LightGBM, and the only model besides LightGBM to clear the coverage bar. It also **loses badly on
decision cost**: 18% more expensive than LightGBM, 13% more expensive than plain SeasonalNaive.
Per this project's own Phase 7 acceptance bar ("keep only if it wins on decision cost"), **NBEATS
is not adopted** — LightGBM stays the production model. Worth being direct about this rather than
quietly shelving the result: the headline model of the "stretch" phase lost, and that's the more
useful finding, not a disappointing one.

**Why it loses, and what it reveals.** The holding/stockout split is the tell: NBEATS holds 30%
less inventory and stocks out far more than SeasonalNaive (fill rate 71.5% vs. 83.2%). This is the
same directional pattern as Phase 4's original LightGBM finding — a more accurate model ending up
*leaner*, not more conservative — but it happened even with the empirical-residual safety-stock
fix already in place, which was built specifically to prevent that class of bug. The likely
mechanism this time is different: on a 77%-zero-sale-day panel, seasonal naive repeats last week's
value forward, so any series with a recent nonzero sale gets that value echoed across the whole
horizon — inflating its average forecast (and thus its reorder point) in a way that shows up as
*worse accuracy* but functions as an *unpriced safety margin* against demand spikes. NBEATS, being
more accurate, doesn't carry that accidental cushion. And it's not NBEATS-specific: fill rate
across every model this project has evaluated — LightGBM 81.5%, AutoETS 80.7%, AutoTheta 81.1%,
MovingAverage 80.7%, SeasonalNaive 83.2%, NBEATS 71.5% — falls short of the 95% service-level
target. That every model misses, in a narrow band except for the most accurate one, points to the
decision layer's safety-stock formula itself: a symmetric normal approximation (`z · σ`) built
from empirical residual variance likely understates the true risk of a right-skewed, spiky demand
distribution, regardless of how accurately σ is measured. A skewed or empirical-quantile-based
safety-stock formula, rather than the current normal approximation, is the natural next step —
flagged here as a real, evidence-backed limitation rather than fixed under time pressure to force
a cleaner headline number.

**Also a scoping note, not just a result:** this NBEATS integration is univariate — no calendar,
price, or SNAP features, unlike LightGBM's feature pipeline. Extending it to consume the same
exogenous signals is a real feature-engineering project of its own, deliberately out of scope for
a stretch phase (see `docs/bridges2.md`). Some of the accuracy gap, and possibly some of the cost
gap, could reflect that feature disparity rather than architecture alone.
