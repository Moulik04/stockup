# Production model — what LightGBM was for, what it costs, and what replaced it — 2026-09-29

Shipped policy throughout: `S = s + 2 x lead-time demand`, 95% target, legacy (normal, intermittency) safety-stock scheme, 25%/yr holding, 27.5% lost margin. Same 400-series sample, 4 folds, and forecast/calibration caches as `reports/croston_*.md`.

## 1. Do the models' own P10/P90 feed the reorder decision?

The shipped cell was simulated twice over 12,800 model×fold×series rows: once on the real forecasts, once with every model's P10 set to 0 and P90 to 1,000 (P50 untouched). Largest absolute change per column:

| column | max |real − scrambled| | role |
| --- | --- | --- |
| reorder_point | 0 | decision |
| order_up_to | 0 | decision |
| total_cost | 0 | decision |
| lead_time_std | 1032.25 | diagnostic (spread column) |

**No. Reorder points, order-up-to levels and cost are unchanged to the last bit; the only column that moves is the quantile-derived spread, which is recorded but not read by sizing.** Sizing is P50 lead-time demand + a buffer from pooled empirical residuals; the order-up-to level is P50-derived too. The quantiles are read only as a fallback for a series the calibration has no buffer for, which does not occur on this panel.

## 2. Cost against LightGBM, paired bootstrap (95% CI, series resampled)

Negative = cheaper than LightGBM. Six candidates plus SeasonalNaive as the reference that genuinely differs. Across six candidates, at least one CI missing zero by chance alone is roughly a one-in-four event, so a lone marginal result would not be evidence.

### All series

| model | cost / fold | minus LightGBM | 95% CI | reading |
| --- | --- | --- | --- | --- |
| MovingAverage | $438 | $+11 | [$-34, $+51] | tied |
| AutoETS | $425 | $-2 | [$-51, $+39] | tied |
| AutoTheta | $449 | $+22 | [$-32, $+70] | tied |
| CrostonClassic | $442 | $+15 | [$-27, $+52] | tied |
| CrostonSBA | $441 | $+14 | [$-42, $+59] | tied |
| TSB | $452 | $+26 | [$-21, $+69] | tied |
| SeasonalNaive | $526 | $+99 | [$+34, $+160] | LightGBM cheaper |

### Intermittent series (>50% zero training days)

| model | cost / fold | minus LightGBM | 95% CI | reading |
| --- | --- | --- | --- | --- |
| MovingAverage | $371 | $+13 | [$-35, $+53] | tied |
| AutoETS | $358 | $-0 | [$-51, $+41] | tied |
| AutoTheta | $382 | $+24 | [$-31, $+71] | tied |
| CrostonClassic | $374 | $+16 | [$-25, $+53] | tied |
| CrostonSBA | $374 | $+16 | [$-40, $+61] | tied |
| TSB | $386 | $+28 | [$-17, $+70] | tied |
| SeasonalNaive | $447 | $+89 | [$+22, $+154] | LightGBM cheaper |

### Regular series

| model | cost / fold | minus LightGBM | 95% CI | reading |
| --- | --- | --- | --- | --- |
| MovingAverage | $66 | $-2 | [$-5, $+1] | tied |
| AutoETS | $67 | $-2 | [$-5, $+2] | tied |
| AutoTheta | $67 | $-2 | [$-5, $+2] | tied |
| CrostonClassic | $68 | $-1 | [$-5, $+3] | tied |
| CrostonSBA | $67 | $-2 | [$-7, $+3] | tied |
| TSB | $67 | $-2 | [$-6, $+2] | tied |
| SeasonalNaive | $78 | $+10 | [$+3, $+18] | LightGBM cheaper |

## 3. What each model costs to run

Fit + predict wall time on the last fold's training window (400 series, one machine, all cores). The statistical models fit per series, so their time scales roughly linearly in series count; LightGBM is one global fit. `coverage_80` is the share of test days inside the model's own P10–P90 band — what `/forecast` and the dashboard fan chart showed through v1.1, before they switched to the calibrated band of section 4.

| model | fit_predict_s | coverage_80 |
| --- | --- | --- |
| LightGBM | 22.9 | 81.2% |
| MovingAverage | 3.2 | 62.2% |
| AutoETS | 102.8 | 63.6% |
| AutoTheta | 224.0 | 62.9% |
| CrostonClassic | 4.8 | 58.7% |
| CrostonSBA | 4.5 | 59.4% |
| TSB | 4.4 | 62.3% |
| TrailingMean (served) | 0.1 | n/a (no native band) |

Not measured here, read from the code: LightGBM needs `features.build_features` (lags, rolling stats, calendar, price, event flags) and a forward exog proxy at serve time (`exog.py`, repeat-the-recent-pattern), and persists a ~110 MB artifact; the statistical and Croston models need only each series' own demand history and persist nothing. The `TrailingMean (served)` row is the implementation `train.py` now fits: the same forecast as MovingAverage with no statsforecast call. Timing note: it is measured here on the 400-series fold like every other row; on the full 5,650-series panel it takes about two seconds.

## 4. Does the interval `/forecast` shows match the decision?

`/forecast` and the dashboard's fan chart now show `P50 ± z(80% central) × σ ÷ √7` per day, where σ is the same pooled residual std the reorder point's buffer is `z(95%) × σ` of (`SafetyStockCalibration.calibrated_interval`). Evaluated out of sample on all four folds with the calibration held to fold 1's residual window, through the production code path:

| model | native daily coverage | calibrated daily coverage | native / calibrated daily width | lead-time total coverage | missed below / above | reorder point covers (95% target) |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 81.2% | 91.0% | 1.68 / 2.34 | 87.0% | 5.3% / 7.7% | 94.8% |
| MovingAverage | 62.2% | 91.4% | 1.38 / 2.37 | 87.1% | 6.1% / 6.8% | 95.1% |

Nominal coverage is 80%. Read it three ways. **The reorder point does what it says**: realised lead-time demand stayed at or under it in 94.8%–95.1% of 6,400 held-out windows against a 95% target (in-stock probability, not the volume-weighted fill rate the cost tables report). **The band over-covers, which is the safe direction to be wrong in**: the lower edge is clipped at zero and most demand is zero, so any band as wide as the reorder point's spread contains nearly every zero day; the independent-days assumption in `σ ÷ √L` may add width too, which this table cannot separate. **The model's own quantiles do not describe the decision**: they cover 62%–81% of days depending on the model, and LightGBM's near-nominal figure comes from alphas tuned to hit that coverage (`DECISIONS.md`, 2026-09-03), not from anything about the buffer stocked. The calibrated band depends on the model only through its residual spread, so what is shown follows what is decided.

## 5. Is the served model the evaluated one?

`TrailingMeanModel` against the ladder's MovingAverage on all 4 folds' training windows: same (series, date) rows, largest P50 difference 4.1e-07 (statsforecast computes in float32). The cost comparison in section 2 therefore applies to the served model as it stands.
