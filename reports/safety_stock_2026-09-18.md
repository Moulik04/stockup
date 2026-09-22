# Safety-stock calibration: ablation — 2026-09-18

Two independent levers behind the decision layer's safety stock, measured separately and together, plus a fifth cell testing a cleaner alternative bucketing scheme. Same folds, models, sample and simulation as `reports/decision_<date>.md`; the only thing that varies is how a bucket of forecast residuals becomes a buffer.

- **Form** — `normal`: `z · σ_pooled`, assuming symmetric normal lead-time demand. `empirical`: the residual distribution's own quantile at the target service level, assuming nothing about its shape.
- **Granularity** — `intermittency`: the current 2 buckets. `volume_tercile_within_intermittency` (renamed from `intermittency_volume` — see below for why): crossed with volume terciles, 6 theoretical buckets. `volume_quintile` (fifth cell, `normal` form only): volume alone, 5 balanced buckets, no intermittency dimension — the same information this panel's volume and intermittency both carry, split into more, evenly populated groups.

Target service level 95%. Cost, fill rate and CSL are means across folds and models.

## The ablation cells

| form | granularity | mean_total_cost | holding_cost | stockout_cost | fill_rate | cycle_service_level |
| --- | --- | --- | --- | --- | --- | --- |
| normal | volume_quintile | 13223.213 | 6395.545 | 6827.667 | 0.827 | 0.801 |
| normal | volume_tercile_within_intermittency | 13458.397 | 6547.431 | 6910.966 | 0.825 | 0.807 |
| empirical | volume_tercile_within_intermittency | 13547.512 | 6509.163 | 7038.350 | 0.822 | 0.798 |
| empirical | intermittency | 14662.064 | 6660.203 | 8001.861 | 0.797 | 0.799 |
| normal | intermittency | 14847.658 | 7519.438 | 7328.219 | 0.815 | 0.828 |

Baseline (the shipped scheme) is **normal × intermittency**: $14,848 per fold, fill rate 81.5%, CSL 82.8%. Best cell by cost is **normal × volume_quintile** at $13,223 (-10.9%), fill rate 82.7%, CSL 80.1%.

## Per model

| form | granularity | model | total_cost | fill_rate | cycle_service_level |
| --- | --- | --- | --- | --- | --- |
| empirical | intermittency | AutoETS | 14513.106 | 0.796 | 0.786 |
| empirical | intermittency | AutoTheta | 14667.442 | 0.790 | 0.788 |
| empirical | intermittency | LightGBM | 14489.054 | 0.797 | 0.796 |
| empirical | intermittency | MovingAverage | 14628.474 | 0.798 | 0.796 |
| empirical | intermittency | SeasonalNaive | 15012.245 | 0.806 | 0.831 |
| empirical | volume_tercile_within_intermittency | AutoETS | 13423.458 | 0.826 | 0.797 |
| empirical | volume_tercile_within_intermittency | AutoTheta | 13593.011 | 0.819 | 0.792 |
| empirical | volume_tercile_within_intermittency | LightGBM | 13467.373 | 0.816 | 0.779 |
| empirical | volume_tercile_within_intermittency | MovingAverage | 13597.637 | 0.821 | 0.792 |
| empirical | volume_tercile_within_intermittency | SeasonalNaive | 13656.082 | 0.828 | 0.828 |
| normal | intermittency | AutoETS | 14726.361 | 0.807 | 0.819 |
| normal | intermittency | AutoTheta | 14830.457 | 0.812 | 0.825 |
| normal | intermittency | LightGBM | 14543.379 | 0.815 | 0.826 |
| normal | intermittency | MovingAverage | 14914.193 | 0.807 | 0.823 |
| normal | intermittency | SeasonalNaive | 15223.897 | 0.832 | 0.847 |
| normal | volume_quintile | AutoETS | 13219.520 | 0.818 | 0.785 |
| normal | volume_quintile | AutoTheta | 13195.672 | 0.825 | 0.800 |
| normal | volume_quintile | LightGBM | 12991.909 | 0.825 | 0.791 |
| normal | volume_quintile | MovingAverage | 13344.030 | 0.820 | 0.795 |
| normal | volume_quintile | SeasonalNaive | 13364.931 | 0.847 | 0.832 |
| normal | volume_tercile_within_intermittency | AutoETS | 13424.307 | 0.817 | 0.794 |
| normal | volume_tercile_within_intermittency | AutoTheta | 13455.045 | 0.823 | 0.802 |
| normal | volume_tercile_within_intermittency | LightGBM | 13216.660 | 0.823 | 0.795 |
| normal | volume_tercile_within_intermittency | MovingAverage | 13573.600 | 0.818 | 0.808 |
| normal | volume_tercile_within_intermittency | SeasonalNaive | 13622.374 | 0.845 | 0.836 |

## Can the buckets support the estimate?

The constraint named up front: ~400 residuals per fold, split two ways, leaves ~200 per bucket — a 95% quantile then rests on ~10 tail observations, and a six-way split was expected to cut that to ~3. It turned out less severe than that (the six-way split only populates four buckets — see below), but still thin. Each bucket's statistic carries a bootstrap CI (2,000 resamples) on the quantile itself so the thinness is visible rather than implied.

Median bucket sizes and CI widths (CI width as a fraction of the point estimate):

| form | granularity | median_n | min_n | median_ci_width_frac | max_ci_width_frac |
| --- | --- | --- | --- | --- | --- |
| empirical | intermittency | 200.000 | 40.000 | 0.859 | 2.664 |
| empirical | volume_tercile_within_intermittency | 113.000 | 40.000 | 0.851 | 2.664 |
| normal | intermittency | 200.000 | 40.000 | 0.480 | 0.963 |
| normal | volume_quintile | 80.000 | 80.000 | 0.471 | 0.839 |
| normal | volume_tercile_within_intermittency | 113.000 | 40.000 | 0.437 | 0.963 |

**Why the rename.** The scheme originally called `intermittency_volume` only ever populates 4 of its 6 theoretical buckets, not 6, because intermittency and volume are close to the same variable on this panel: every one of the ~42 `regular` series lands in the top volume tercile, so `regular_lo` and `regular_mid` are empty and the `regular` class is never actually subdivided. What it actually does is split the *intermittent* class three ways by volume — so it is now named `volume_tercile_within_intermittency`, describing that behaviour instead of the clean 2x3 cross the old name implied. The fifth cell, `volume_quintile`, pools by volume alone and populates all 5 of its buckets — a cleaner alternative carrying the same information (volume already implies intermittency here) in more, evenly sized groups, with no collapsed cells to explain away.

The empirical quantile's CI spans a median 85% of the estimate itself at the finer granularity and 86% at the coarser one — the split barely moves it, because the binding constraint is the *tail* count, and the smallest bucket (40 residuals) is the same `regular` group in both. The same buckets sized by `z · std` carry a median CI of 44% — roughly half as wide, because a standard deviation uses every observation while a 95th percentile leans on the handful above it.

**That is a data-volume finding, not a distributional one, and it changes how lever A should be stated.** A CI spanning 85% of its own point estimate against 44% for the alternative means the empirical buffer is pinned down far less tightly than the normal one, so the comparison between them is **underpowered at this sample size** — the bootstrap section below shows what that does to the cost difference. It is not evidence that empirical quantiles "lose": the estimator's own precision, not the shape of the distribution it targets, is the binding constraint: bringing the empirical form's CI down to the normal form's 44% width would take roughly 3.8x as many residuals per bucket as the ~113 available today (≈427, under the standard bootstrap-CI-width ∝ 1/√n scaling). That is a real, quantifiable, and fixable gap — more series per bucket, not a different estimator — and it is the reason a real budget ("What I'd do with a budget") should spend on more series before spending on a fancier form.

## Diagnostic 1 — fill rate against volume decile

Run on the current scheme (normal × intermittency). If pooling in absolute units is what depresses service, fill rate should fall as volume rises: the same buffer in units covers a low-volume SKU comfortably and a high-volume one barely.

Pooled across models, fill rate declines from 97.7% in the bottom three volume deciles to 78.8% in the top three. A declining relationship is the signature of pooling in absolute units: one buffer in units is generous for a low-volume SKU and thin for a high-volume one.

![Fill rate by volume decile](safety_stock_files/fill_rate_by_volume_2026-09-18.png)

## Diagnostic 2 — within-bucket dispersion of per-series residual std

If a single bucket spans an order of magnitude in per-series residual std, pooling it into one absolute buffer is indefensible regardless of what the cost table says.

| bucket | n | p10 | median | p90 | max | p90_over_p10 |
| --- | --- | --- | --- | --- | --- | --- |
| intermittent | 360.000 | 0.504 | 1.465 | 3.887 | 15.755 | 7.716 |
| regular | 40.000 | 1.688 | 4.532 | 13.880 | 31.235 | 8.224 |

## Bootstrap confidence intervals (v1.1 Task 3, calibration cells)

Every point estimate above rests on one 400-series/4-fold run. A paired bootstrap over series (2,000 resamples, series drawn with replacement, folds and models pooled exactly as the summary table pools them — see `bootstrap.py`) puts a 95% CI on the *differences* that matter: every pairwise cost gap among the four `{normal, empirical} x {intermittency, volume_tercile_within_intermittency}` cells, the fifth cell against the shipped baseline, and — the sharper question — whether the cost win and the CSL regression are both real.

### Pairwise cost differences, the four 2x2 cells

| cell_a | cell_b | mean_cost_diff | ci_lo | ci_hi | crosses_zero |
| --- | --- | --- | --- | --- | --- |
| normal×intermittency | empirical×intermittency | 185.593 | 52.725 | 323.925 | False |
| normal×intermittency | normal×volume_tercile_within_intermittency | 1389.260 | 1074.419 | 1726.535 | False |
| normal×intermittency | empirical×volume_tercile_within_intermittency | 1300.145 | 937.658 | 1688.506 | False |
| empirical×intermittency | normal×volume_tercile_within_intermittency | 1203.667 | 897.346 | 1522.748 | False |
| empirical×intermittency | empirical×volume_tercile_within_intermittency | 1114.552 | 769.402 | 1479.354 | False |
| normal×volume_tercile_within_intermittency | empirical×volume_tercile_within_intermittency | -89.115 | -181.162 | 2.807 | True |

`mean_cost_diff` = cell_a − cell_b, per fold; a negative value means cell_a is cheaper. `crosses_zero` = the 95% CI includes 0, i.e. the two cells' cost is not distinguishable at this sample size.

### Fifth cell

`volume_quintile` (normal form) vs. the shipped `normal × intermittency`: cost difference $-1,624.4 [$-2,191.1, $-1,091.1] per fold.

`volume_quintile` vs. `normal × volume_tercile_within_intermittency` (the scheme it is meant to replace): $-235.2 [$-615.4, $131.9] per fold — not distinguishable from it at this sample size.

### The cost win vs. the CSL regression — best cell (`normal × volume_quintile`) vs. baseline

- **Cost**: $-1,624.4 [$-2,191.1, $-1,091.1] per fold (does not cross zero — a real saving).
- **Cycle service level**: -2.7pp [-4.1pp, -1.3pp] (does not cross zero — a real regression).
- **Fill rate**: +1.2pp [-0.1pp, 2.7pp] (crosses zero).

**Both the cost win and the CSL regression are statistically real.** This is a genuine trade-off, not a free lunch: adopting the finer scheme buys a real cost saving at the price of a real service-level cost *at the same nominal target*. That is not the end of it: the two schemes realise different service at the same target, so the fair comparison is at matched realised service — see `reports/service_level_sweep_<date>.md` §5, where the finer scheme is still cheaper with no distinguishable CSL loss. Whether to change the shipped default is MJ's call, not one to flip silently.

## What the ablation says

**Granularity is the lever that moves cost; distributional form is not.** Splitting the buckets alone (normal × volume_tercile_within_intermittency) takes cost from $14,848 to $13,458 (-9.4%) and lifts fill rate 81.5% → 82.5%. Swapping the form alone (empirical × intermittency) moves cost to $14,662 (-1.2%) and *lowers* fill rate to 79.7%. Doing both ($13,548) is no better than granularity alone. Lever A (form), from the bootstrap: at the coarser granularity the empirical form is cheaper by $186 per fold (95% CI $53 to $324); at the finer granularity the two forms differ by $89 per fold (95% CI $-181 to $3) — not distinguishable at this sample size. That is a small, granularity-dependent effect next to granularity's own, measured with an estimator whose bucket CI spans roughly twice the relative width of `z · std`'s (see above) — an underpowered comparison at ~10 tail points per bucket. It is a *data-volume* limit on what a tail quantile can deliver here, not a finding about empirical quantiles or about the demand distribution.

The fifth cell, `normal × volume_quintile`, lands at $13,223 (-10.9% vs. baseline, -1.7% vs. `volume_tercile_within_intermittency`; the gap to it is not distinguishable from zero at this sample size). Pure volume pooling populates every bucket and does not lean on the two variables being near-duplicates, so it is the cleaner scheme to keep even where it only ties.

Both diagnostics predicted the granularity result. Fill rate falls steeply across volume deciles, and per-series residual std spans nearly an order of magnitude inside a single bucket (p90/p10 ≈ 8). Pooling in absolute units was the defect; the distribution's shape was not the binding one.

**Neither lever closes the service-level gap.** The best cell on service reaches 82.7% fill rate against a 95% target — still 12 percentage points short — and cycle service level actually *falls* from 82.8% to 80.1% in the lowest-cost cell (`normal × volume_quintile`) as cost improves. That divergence is informative rather than contradictory: finer buckets move buffer from over-provisioned low-volume SKUs to under-provisioned high-volume ones. Fill rate is unit-weighted, so it improves; CSL is cycle-weighted, and low-volume series generate a disproportionate share of cycles, so it slips. A paired bootstrap over series shows both the cost saving *and* the CSL regression are real (neither CI includes zero) — this is a genuine trade-off, not noise in either direction; see "Bootstrap confidence intervals" above.

**So the 12-point service gap is not distributional, and not mainly about pooling either.** Better pooling buys real cost savings, but leaves the service gap roughly where it was. Having now ruled out both the shape of the residual distribution and the granularity it is estimated at, the remaining suspect is the policy's structure rather than its calibration: `S = s` (docs/decision.md step 3) means every order is only the accumulated undershoot, so inventory is rebuilt to the reorder point and no further. A policy that never orders more than it is short cannot hold a service buffer against the next cycle, no matter how well that buffer is sized. That is the next thing to test. See `reports/service_level_sweep_<date>.md` (v1.1 Task 7) for whether the 95% target itself is where the project's own costs say it should be — run before treating any of this cell's numbers as the new default.
