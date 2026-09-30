# Track B: do the findings replicate at a small UK online gift-ware seller? 2026-09-30

UCI Online Retail II (Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D, CC BY 4.0), cleaned to weekly SKU series (`docs/data.md`). The design, the economics and the rule for each verdict were registered before any model ran (`docs/track_b.md`, public commits 2026-09-29); nothing below is re-read after the fact. **Verdicts are the primary panel's** (1,742 SKUs sold in at least 26 of the first 52 weeks); the robustness panel (2,666 SKUs, at least 13) is reported as agrees or disagrees and never used to pick the more favourable result.

## Which findings replicated

**0 of 4 replicated on the primary panel.** The robustness panel agrees on all four.

| finding | Track A | Track B primary panel | robustness panel | primary verdict across the 12 economics cells |
| --- | --- | --- | --- | --- |
| 1. The ordering policy matters more than the model | saving $1,127; cluster spread $27 | **not replicated**: S>s saves $20,589 per fold per model; cluster spread $14,495; every CI excludes zero: False | agrees | 7 of 12 |
| 2. The calibrated quantile does what it says; the shortfall is the policy | coverage 94.8%-95.1%; CSL 91.7%-93.4% | **not replicated**: coverage 98.1% (within 3 pts for every cluster model: False); realised CSL 94.7%; undershoot 73% of stockouts | agrees | 12 of 12 |
| 3. The model cluster is a tie on cost | all six CIs against LightGBM contain zero | **not replicated**: 13 of 21 pairs exclude zero | agrees | 12 of 12 |
| 4. The trailing mean is enough | MovingAverage minus LightGBM $11 [-$34, +$51] | **not replicated**: MovingAverage minus LightGBM $7,803 [$4,770, $10,536] | agrees | 8 of 12 |

Default cell: margin 27.5%, lead time 2 weeks, 95% target. Track A's column is from the frozen baseline, not restated. Read with the coverage limits in `docs/track_b.md`: both panels are products with at least a year of history, both decline over the test period, and the primary panel tests mature products with muted seasonality.

## Finding 1 — policy against model

Primary panel, cost per fold (USD, 1,742 series). `*` SeasonalNaive is in the ladder but not the cluster.

| model | S = s, cost/fold | S > s, cost/fold | S>s minus S=s | CSL (S>s) | fill rate (S>s) |
| --- | --- | --- | --- | --- | --- |
| SeasonalNaive * | $221,291 | $218,821 | -$2,470 [-$9,887, $4,141] | 96.3% | 94.0% |
| MovingAverage | $200,656 | $182,357 | -$18,299 [-$26,549, -$11,464] | 95.6% | 94.0% |
| AutoETS | $198,862 | $177,504 | -$21,358 [-$30,203, -$13,902] | 95.2% | 94.4% |
| AutoTheta | $196,740 | $173,822 | -$22,918 [-$31,586, -$15,606] | 95.3% | 94.1% |
| LightGBM | $198,359 | $174,555 | -$23,805 [-$32,724, -$16,458] | 94.7% | 93.7% |
| CrostonClassic | $197,794 | $174,163 | -$23,631 [-$32,339, -$16,073] | 94.8% | 94.1% |
| CrostonSBA | $200,067 | $173,541 | -$26,526 [-$35,744, -$18,580] | 94.6% | 93.8% |
| TSB | $193,564 | $167,862 | -$25,702 [-$34,604, -$18,126] | 94.7% | 94.1% |

Robustness panel (2,666 series):

| model | S = s, cost/fold | S > s, cost/fold | S>s minus S=s | CSL (S>s) | fill rate (S>s) |
| --- | --- | --- | --- | --- | --- |
| SeasonalNaive * | $378,659 | $376,279 | -$2,380 [-$10,209, $4,290] | 95.3% | 92.0% |
| MovingAverage | $263,279 | $242,246 | -$21,033 [-$29,309, -$13,565] | 94.9% | 92.3% |
| AutoETS | $266,872 | $243,459 | -$23,413 [-$32,172, -$15,626] | 94.7% | 92.9% |
| AutoTheta | $260,633 | $235,040 | -$25,592 [-$34,562, -$17,739] | 94.6% | 92.5% |
| LightGBM | $259,550 | $232,342 | -$27,208 [-$35,932, -$19,589] | 93.8% | 92.2% |
| CrostonClassic | $290,312 | $266,195 | -$24,117 [-$33,625, -$15,996] | 94.9% | 93.0% |
| CrostonSBA | $291,047 | $263,688 | -$27,359 [-$37,224, -$18,928] | 94.7% | 92.7% |
| TSB | $257,739 | $227,759 | -$29,980 [-$39,451, -$21,788] | 94.1% | 92.5% |

## Finding 2 — calibrated quantile against undershoot

Held-out coverage of the reorder point on fixed-origin lead-time windows inside the test folds, against realised cycle service level under `S > s`, primary panel:

| lead time | mean coverage | range across models | realised CSL (S>s) | gap | undershoot share | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 wk | 98.5% | 98.5%-98.7% | 100.0% | -1.5 pts | n/a | not replicated |
| 2 wk | 98.1% | 97.9%-98.3% | 94.7% | 3.3 pts | 73% | not replicated |
| 4 wk | 98.6% | 98.4%-98.6% | 94.3% | 4.2 pts | 55% | not replicated |

Robustness panel:

| lead time | mean coverage | range across models | realised CSL (S>s) | gap | undershoot share | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 wk | 98.6% | 98.4%-98.8% | 100.0% | -1.4 pts | n/a | not replicated |
| 2 wk | 98.3% | 98.0%-98.5% | 94.1% | 4.2 pts | 69% | not replicated |
| 4 wk | 98.7% | 98.4%-98.9% | 93.6% | 5.1 pts | 48% | not replicated |

## Findings 3 and 4 — the cluster and the trailing mean

Each cluster model against LightGBM under `S > s` (paired bootstrap, series resampled), primary panel:

| model | cost/fold (S>s) | minus LightGBM | reading |
| --- | --- | --- | --- |
| AutoETS | $177,504 | $2,950 [-$1,249, $6,313] | tied |
| AutoTheta | $173,822 | -$733 [-$4,494, $2,367] | tied |
| CrostonClassic | $174,163 | -$392 [-$3,332, $2,207] | tied |
| CrostonSBA | $173,541 | -$1,013 [-$3,952, $1,677] | tied |
| MovingAverage | $182,357 | $7,803 [$4,770, $10,536] | dearer |
| TSB | $167,862 | -$6,693 [-$9,400, -$4,389] | cheaper |

Of the 21 pairs among the seven cluster models, 13 have a CI that excludes zero: AutoETS - AutoTheta $3,683 [$1,959, $5,255]; AutoETS - CrostonSBA $3,963 [$440, $6,998]; AutoETS - TSB $9,643 [$6,445, $12,287]; AutoTheta - TSB $5,960 [$3,402, $8,082]; CrostonClassic - TSB $6,301 [$5,065, $7,574]; CrostonSBA - TSB $5,679 [$4,035, $7,350]; LightGBM - TSB $6,693 [$4,389, $9,400]; MovingAverage - AutoETS $4,853 [$1,307, $9,288]; MovingAverage - AutoTheta $8,536 [$5,558, $12,414]; MovingAverage - CrostonClassic $8,194 [$4,798, $11,574]; MovingAverage - CrostonSBA $8,816 [$5,153, $12,412]; MovingAverage - LightGBM $7,803 [$4,770, $10,536]; MovingAverage - TSB $14,495 [$11,392, $17,679]. With 21 comparisons about one exclusion is expected by chance alone even among models that are truly tied.

Robustness panel:

| model | cost/fold (S>s) | minus LightGBM | reading |
| --- | --- | --- | --- |
| AutoETS | $243,459 | $11,117 [$6,815, $14,656] | dearer |
| AutoTheta | $235,040 | $2,698 [-$1,237, $5,933] | tied |
| CrostonClassic | $266,195 | $33,852 [$28,803, $39,040] | dearer |
| CrostonSBA | $263,688 | $31,345 [$26,564, $36,304] | dearer |
| MovingAverage | $242,246 | $9,903 [$6,911, $12,994] | dearer |
| TSB | $227,759 | -$4,584 [-$7,691, -$1,710] | cheaper |

## Accuracy against cost

Primary panel. Rank correlation between the MASE ranking and the cost ranking: 0.38.

| model | MASE | rank by MASE | rank by cost (S>s) | native 80% band coverage |
| --- | --- | --- | --- | --- |
| SeasonalNaive | 1.225 | 7 | 8 | 62.8% |
| MovingAverage | 0.893 | 2 | 7 | 68.5% |
| AutoETS | 0.952 | 4 | 6 | 67.7% |
| AutoTheta | 0.909 | 3 | 3 | 70.8% |
| LightGBM | 1.346 | 8 | 5 | 53.9% |
| CrostonClassic | 1.036 | 6 | 4 | 68.1% |
| CrostonSBA | 1.008 | 5 | 2 | 68.3% |
| TSB | 0.887 | 1 | 1 | 70.6% |

Robustness panel: rank correlation 0.48.

| model | MASE | rank by MASE | rank by cost (S>s) | native 80% band coverage |
| --- | --- | --- | --- | --- |
| SeasonalNaive | 1.293 | 7 | 8 | 65.7% |
| MovingAverage | 0.979 | 2 | 4 | 70.2% |
| AutoETS | 1.062 | 4 | 5 | 69.5% |
| AutoTheta | 1.006 | 3 | 3 | 72.2% |
| LightGBM | 1.839 | 8 | 2 | 56.4% |
| CrostonClassic | 1.238 | 6 | 7 | 70.0% |
| CrostonSBA | 1.206 | 5 | 6 | 70.2% |
| TSB | 0.979 | 1 | 1 | 72.2% |

## Sensitivity to the assumed economics

`R` marks a replicated verdict in that cell of the margin x lead-time grid (finding 2 depends on the lead time only). Primary panel:

| margin | lead time | finding 1 | finding 2 | finding 3 | finding 4 |
| --- | --- | --- | --- | --- | --- |
| 20.0% | 1 wk | - | - | - | - |
| 20.0% | 2 wk | - | - | - | - |
| 20.0% | 4 wk | - | - | - | - |
| 27.5% | 1 wk | - | - | - | - |
| 27.5% | 2 wk | - | - | - | - |
| 27.5% | 4 wk | - | - | - | - |
| 40.0% | 1 wk | R | - | - | R |
| 40.0% | 2 wk | R | - | - | R |
| 40.0% | 4 wk | - | - | - | - |
| 50.0% | 1 wk | R | - | - | R |
| 50.0% | 2 wk | R | - | - | R |
| 50.0% | 4 wk | R | - | - | - |

Robustness panel:

| margin | lead time | finding 1 | finding 2 | finding 3 | finding 4 |
| --- | --- | --- | --- | --- | --- |
| 20.0% | 1 wk | - | - | - | - |
| 20.0% | 2 wk | - | - | - | - |
| 20.0% | 4 wk | - | - | - | - |
| 27.5% | 1 wk | - | - | - | - |
| 27.5% | 2 wk | - | - | - | - |
| 27.5% | 4 wk | - | - | - | - |
| 40.0% | 1 wk | R | - | - | R |
| 40.0% | 2 wk | R | - | - | - |
| 40.0% | 4 wk | - | - | - | - |
| 50.0% | 1 wk | R | - | - | R |
| 50.0% | 2 wk | R | - | - | R |
| 50.0% | 4 wk | R | - | - | - |

## The 924 SKUs only the robustness panel holds

Descriptive, no verdict: the short-season end of the catalogue (`docs/track_b.md`). Cost per fold under `S > s`, default cell, and MASE.

| model | cost/fold (S>s) | CSL | MASE |
| --- | --- | --- | --- |
| SeasonalNaive | $144,760 | 93.0% | 1.423 |
| MovingAverage | $60,296 | 93.0% | 1.141 |
| AutoETS | $66,813 | 93.6% | 1.269 |
| AutoTheta | $62,055 | 92.8% | 1.190 |
| LightGBM | $58,541 | 91.9% | 2.902 |
| CrostonClassic | $90,801 | 95.1% | 1.620 |
| CrostonSBA | $89,012 | 94.9% | 1.578 |
| TSB | $60,878 | 92.6% | 1.150 |

## The calibration window, descriptively

Post hoc and descriptive; it changes no verdict. Every model's buffer is sized from one window, the last two weeks of the first 52, which is late November: the autumn peak. That window's demand level, each model's sigma from it, and each model's forecast bias:

Primary panel: the calibration window (2010-11-22 to 2010-11-29) averaged 63.4 units per SKU-week; the test folds averaged fold 1: 30.9, fold 2: 34.0, fold 3: 32.3, fold 4: 42.8. Rank correlation between a model's calibration sigma and its cost rank: 0.76.

| model | calibration sigma (units per lead time) | forecast bias (mean p50 / mean actual - 1) |
| --- | --- | --- |
| SeasonalNaive | 220 | +37% |
| MovingAverage | 182 | +17% |
| AutoETS | 183 | +28% |
| AutoTheta | 173 | +21% |
| LightGBM | 161 | +46% |
| CrostonClassic | 167 | +23% |
| CrostonSBA | 170 | +17% |
| TSB | 160 | +18% |

Robustness panel: the calibration window (2010-11-22 to 2010-11-29) averaged 49.2 units per SKU-week; the test folds averaged fold 1: 23.1, fold 2: 24.7, fold 3: 24.1, fold 4: 34.3. Rank correlation between a model's calibration sigma and its cost rank: 1.00.

| model | calibration sigma (units per lead time) | forecast bias (mean p50 / mean actual - 1) |
| --- | --- | --- |
| SeasonalNaive | 184 | +36% |
| MovingAverage | 153 | +17% |
| AutoETS | 160 | +32% |
| AutoTheta | 151 | +22% |
| LightGBM | 149 | +56% |
| CrostonClassic | 181 | +29% |
| CrostonSBA | 180 | +22% |
| TSB | 147 | +17% |

## Limits that travel with every result above

- One business, two years, one prior holiday season (2010): no model can learn an annual pattern from one cycle beyond repeating it.
- Both panels are products with at least a year of history; cold start is not addressed. Both decline over the test period.
- The calibration is a single lead-time window, the last weeks of November 2010, applied to all four folds.
- SeasonalNaive uses the 52-week season in the folds and the plain naive forecast on the 50-week calibration window; AutoETS and AutoTheta are non-seasonal in every fold.
- Costs, margin and lead time are assumptions; the grid above is the defence, not a proof.
- `y` is units invoiced, not demand.
