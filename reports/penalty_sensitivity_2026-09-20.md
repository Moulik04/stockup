# Penalty and service-target sensitivity of the model ranking — 2026-09-20

Two Track A parameters were unsourced and jointly decide the LightGBM/SeasonalNaive ranking: the holding rate (`reports/holding_breakeven_*.md`) and the stockout penalty, a flat $5.00 per lost unit against a mean item price of $5.91. This report gives the penalty the same treatment, and adds the third quantity the ranking turns out to depend on, the service target.

## 1. The penalty, tied to unit economics

What a retailer loses when it cannot sell a unit is the margin on it, not the price. The default is now a lost sale = **27.5% of price** (Walmart Inc., Form 10-K, fiscal year ended 2026-01-31, Walmart U.S. segment: net sales $482,975M, gross profit $132,615M; 27.2% and 26.8% the two years before). The M5 data are Walmart U.S. stores, so the segment is the matching figure; the consolidated rate (24.2%: net sales $706,413M, cost of sales $535,395M) blends in Sam's Club and International and is a sweep point below. Neither is the hobby category, and Walmart's cost of sales omits some distribution costs. Stock is valued at cost, `(1 - margin) x price`, for the holding rate. At the mean price that is $1.62 per lost unit, against the flat $5.00. It is a **lower bound** on what a stockout costs — lost goodwill and demand that does not return are extra — which is why the margin is swept below rather than trusted.

![Crossover holding rate against the lost-sale margin](penalty_sensitivity_files/crossover_vs_margin.png)

## 2. The crossover surface

Holding 25%/yr for the gap columns; 'cost-optimal' is 99.9999% nominal (`reports/optimal_target_*.md`), held fixed across margins. Positive gap = SeasonalNaive cheaper.

| gross margin | lost sale at mean price | crossover @ 95% | LGB − SN @25%/yr, 95% | cheapest @25%/yr, 95% | crossover @ cost-optimal | LGB − SN @25%/yr, cost-optimal | cheapest @25%/yr, cost-optimal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.0% | $0.30 | 22%/yr | -6 | AutoETS | 5%/yr | -110 | AutoETS |
| 10.0% | $0.59 | 46%/yr | +35 | SeasonalNaive | 10%/yr | -76 | AutoETS |
| 24.2% (Walmart consolidated) | $1.43 | 132%/yr | +151 | SeasonalNaive | 30%/yr | +22 | AutoTheta |
| 27.5% (default, Walmart U.S.) | $1.62 | 157%/yr | +178 | SeasonalNaive | 36%/yr | +44 | AutoTheta |
| 40.0% | $2.36 | 276%/yr | +280 | SeasonalNaive | 63%/yr | +130 | SeasonalNaive |
| 60.0% | $3.55 | 622%/yr | +444 | SeasonalNaive | 141%/yr | +268 | SeasonalNaive |
| 85.0% | $5.02 | 2,350%/yr | +648 | SeasonalNaive | 532%/yr | +439 | SeasonalNaive |
| 95.0% | $5.61 | 7,878%/yr | +730 | SeasonalNaive | 1,783%/yr | +508 | SeasonalNaive |

The closed form `h*(m) ∝ m/(1−m)` holds on the data to a maximum relative error of 9.8e-16 across margins 5%–95%.

## 3. What survives, and what does not

- **At the default 95% target and default economics the crossover is 157%/yr** (5.2× the top of the plausible range), against 364%/yr under the legacy flat $5 penalty. SeasonalNaive is cheaper across the plausible holding range in both — but the '12× outside the range' framing was specific to the flat penalty.
- **At the cost-optimal target it is 36%/yr** — barely above the plausible range's 30% ceiling. Where the gap at 25%/yr and its interval sit there is in `reports/optimal_target_*.md` §3; the point is that the ranking is no longer robust.
- **At margins of ~10% and below the ranking flips inside the plausible holding range** at the cost-optimal target. A lost sale costing under a tenth of the item's price would sit well below Walmart U.S.'s own 27.5% gross margin, so that is the edge of the region, not a likely case.
- **95% target, default economics:** crossover 157%/yr, 95% CI [84%, 243%] (0.0% of resamples had none); share of resamples where LightGBM is cheaper at 15%: 0.0%, 25%: 0.0%, 30%: 0.0%.
- **99.9999% target, default economics:** crossover 36%/yr, 95% CI [10%, 73%] (0.0% of resamples had none); share of resamples where LightGBM is cheaper at 15%: 10.2%, 25%: 31.4%, 30%: 43.4%.

So the defensible statement is not "the naive baseline wins at any real cost" but a conditional one, stated with the intervals at the default economics and 25%/yr (LightGBM minus SeasonalNaive per fold, paired bootstrap):

- default 95% target: +178 [+77, +299]
- cost-optimal 99.9999% target: +44 [-61, +205] — crosses zero

**SeasonalNaive is significantly cheaper at the default target, but the two are statistically indistinguishable at the cost-optimal target.** At the default economics LightGBM is not significantly the cheapest at any plausible holding cost or target shown.
