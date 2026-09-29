# Production model — what LightGBM is for, and what it costs — 2026-09-29

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

Fit + predict wall time on the last fold's training window (400 series, one machine, all cores). The statistical models fit per series, so their time scales roughly linearly in series count; LightGBM is one global fit. `coverage_80` is the share of test days inside the model's own P10–P90 band — what `/forecast` and the dashboard fan chart show, and the only place the models' quantiles still reach a user.

| model | fit_predict_s | coverage_80 |
| --- | --- | --- |
| LightGBM | 22.4 | 81.2% |
| MovingAverage | 3.6 | 62.2% |
| AutoETS | 92.1 | 63.6% |
| AutoTheta | 224.9 | 62.9% |
| CrostonClassic | 4.4 | 58.7% |
| CrostonSBA | 3.6 | 59.4% |
| TSB | 3.8 | 62.3% |

Not measured here, read from the code: LightGBM needs `features.build_features` (lags, rolling stats, calendar, price, event flags) and a forward exog proxy at serve time (`exog.py`, repeat-the-recent-pattern), and persists a ~110 MB artifact; the statistical and Croston models need only each series' own demand history and persist nothing.
