# Backtest report — 2026-09-10

Track A (M5), HOBBIES category, 400-series random subsample (seed=0) of the full ingested panel — keeps `make backtest` a local, iterable command. 4 rolling-origin folds, horizon=28 days, gap=0.

## Summary (mean across folds)

| model | mase | rmsse | wrmsse | pinball_p10 | pinball_p50 | pinball_p90 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NBEATS | 1.108 | 0.865 | 1.125 | 0.071 | 0.304 | 0.242 | 0.845 |
| SeasonalNaive | 1.633 | 1.088 | 1.418 | 0.273 | 0.421 | 0.315 | 0.732 |

## Per-fold breakdown

| fold | model | mase | rmsse | wrmsse | pinball_p10 | pinball_p50 | pinball_p90 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NBEATS | 1.048 | 0.807 | 1.095 | 0.069 | 0.285 | 0.229 | 0.781 |
| 1 | SeasonalNaive | 1.555 | 1.049 | 1.375 | 0.272 | 0.405 | 0.297 | 0.753 |
| 2 | NBEATS | 1.119 | 0.870 | 1.168 | 0.071 | 0.308 | 0.237 | 0.879 |
| 2 | SeasonalNaive | 1.628 | 1.073 | 1.447 | 0.277 | 0.435 | 0.313 | 0.735 |
| 3 | NBEATS | 1.128 | 0.871 | 1.117 | 0.069 | 0.304 | 0.250 | 0.856 |
| 3 | SeasonalNaive | 1.660 | 1.100 | 1.445 | 0.274 | 0.423 | 0.322 | 0.719 |
| 4 | NBEATS | 1.139 | 0.912 | 1.118 | 0.074 | 0.318 | 0.252 | 0.866 |
| 4 | SeasonalNaive | 1.687 | 1.129 | 1.406 | 0.269 | 0.422 | 0.325 | 0.720 |

## Per-department winners (by mean MASE)

| dept_id | best_model | best_mase |
| --- | --- | --- |
| HOBBIES_1 | NBEATS | 1.123 |
| HOBBIES_2 | NBEATS | 1.073 |

## Where the model fails

Best model overall: **NBEATS** (mean MASE = 1.108).

- On series with >50% zero-sale training days (intermittent demand): mean MASE = 1.150 (n=1440)
- On series with <=50% zero-sale training days (regular demand): mean MASE = 0.732 (n=160)

Intermittent series are consistently harder — expected, since a point forecast poorly represents a mostly-zero distribution. This is the strongest argument for the Phase 3 LightGBM model's Tweedie loss over continuing with per-series statistical methods.

## Quantile calibration

**1/2 models miss the ±5-point coverage bar**: SeasonalNaive at 73.2% (nominal target 80%). All of the misses are *under*-covered (intervals too narrow), not over-covered — the conformal calibration used for the statsforecast-based models (2 windows) is the likely culprit there. LightGBM's quantile regressors needed their own calibration fix (nominal alpha narrowed from 0.10/0.90 to 0.13/0.87 after an initial run measured 85.6% coverage) and now clears the bar; see the Phase 3 acceptance check below.
