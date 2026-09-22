# Model-vs-model confidence intervals — 2026-09-18

Paired bootstrap over series (2,000 resamples, seed 0): each draw resamples the 400 series with replacement and recomputes *both* models on the same draw, so the interval is on the difference. Same 400-series / 4-fold sample as every report; CSL policy (`z · σ_pooled`, 2 intermittency buckets — the shipped scheme). `diff` is LightGBM minus the other model; `sign_flips` is the share of resamples whose sign disagrees with the point estimate.

| comparison | metric | diff | ci_lo | ci_hi | sign_flips | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM − SeasonalNaive | cost per fold ($) | -680.518 | -1141.615 | -183.373 | 0.005 | LightGBM better |
| LightGBM − AutoETS | cost per fold ($) | -182.983 | -453.110 | 52.472 | 0.072 | not distinguishable at this sample size |
| LightGBM − AutoTheta | cost per fold ($) | -287.078 | -599.108 | 15.124 | 0.033 | not distinguishable at this sample size |
| LightGBM − MovingAverage | cost per fold ($) | -370.814 | -732.001 | -51.719 | 0.010 | LightGBM better |
| LightGBM − SeasonalNaive | fill rate (pp) | -1.694 | -2.788 | -0.622 | 0.002 | LightGBM worse |
| LightGBM − AutoETS | fill rate (pp) | 0.758 | 0.156 | 1.383 | 0.006 | LightGBM better |
| LightGBM − AutoTheta | fill rate (pp) | 0.307 | -0.513 | 1.081 | 0.205 | not distinguishable at this sample size |
| LightGBM − MovingAverage | fill rate (pp) | 0.768 | -0.026 | 1.591 | 0.028 | not distinguishable at this sample size |
| LightGBM − SeasonalNaive | MASE | -0.045 | -0.100 | 0.011 | 0.059 | not distinguishable at this sample size |
| LightGBM − AutoETS | MASE | 0.057 | 0.034 | 0.082 | 0.000 | LightGBM worse |
| LightGBM − AutoTheta | MASE | 0.064 | 0.034 | 0.096 | 0.000 | LightGBM worse |
| LightGBM − MovingAverage | MASE | 0.057 | 0.031 | 0.087 | 0.000 | LightGBM worse |

MASE rows use the 400 series with a finite scale in every fold (the rest have an all-constant training window); cost and fill-rate rows use all series.

## What survives

**Cost.** LightGBM's cost advantage is distinguishable from zero against SeasonalNaive, MovingAverage, and **not distinguishable** against AutoETS, AutoTheta.

**MASE.** LightGBM is distinguishably *worse* than AutoETS, AutoTheta, MovingAverage and not distinguishable from SeasonalNaive.

Any comparison marked *not distinguishable* is a statement about this sample, not about the models: with 400 series and four folds the gaps between the four clustered models (see `reports/model_divergence_<date>.md`) are smaller than the noise. It should be reported as inconclusive rather than as a ranking.

## Not covered

- **NBEATS vs LightGBM** (accuracy and cost): NBEATS was trained on Bridges-2 and only aggregate tables came back, so there is no per-series detail to resample. `scripts/bridges2/run_deep_backtest.py` now saves it, but recovering it needs a GPU rerun, which needs MJ's approval. Until then the README's NBEATS numbers remain point estimates — the 18%-worse-on-cost gap is large enough that it is unlikely to be noise, but that is an inference, not a measurement.
- **MinTrace vs base**: see `reports/reconciliation_ci_<date>.md`.
