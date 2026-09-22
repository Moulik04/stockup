# Holding-cost breakeven — 2026-09-20

**Under the default cost parameters — holding as a share of unit cost per year, a lost sale costing 27.5% of price (Walmart's gross margin) — LightGBM is the cheapest model only if holding a unit for a year costs more than 1.6× its unit cost (157%/yr), 5.2× the top of the plausible 15%–30%/yr range. At any carrying cost in that range SeasonalNaive is cheaper — and the cheapest of the five models — at the default 95% service target.** Two conditions come with that, and are part of the finding: at the cost-optimal service target (99.9999%) the crossover falls to 36%/yr and the gap is not statistically distinguishable from zero (`reports/optimal_target_*.md`); and under the legacy flat-$5 penalty it was 364%/yr (`reports/penalty_sensitivity_*.md`). The original Phase 4 result had LightGBM cheapest at a holding rate (2%/*day*, 730%/yr) no business pays.

![Cost vs. holding rate: both models' curves, the crossover, and the plausible range](holding_breakeven_files/cost_vs_holding_rate.png)

## 1. The breakeven

Mean simulated cost per 28-day fold, shipped policy (CSL 95%, normal × intermittency), 400 series × 4 folds. Unit economics: a lost sale costs the unit's gross margin and stock is valued at cost, `(1 − margin) × price` (`docs/decision.md`). The stored runs are re-priced exactly per series (`economics.reprice`; see §3).

| holding rate | LightGBM | SeasonalNaive | LightGBM − SeasonalNaive | cheapest of five |
| --- | --- | --- | --- | --- |
| 15%/yr | $1,569 | $1,377 | +192 | SeasonalNaive |
| 20%/yr | $1,604 | $1,419 | +185 | SeasonalNaive |
| 25%/yr | $1,640 | $1,462 | +178 | SeasonalNaive |
| 30%/yr | $1,676 | $1,505 | +171 | SeasonalNaive |
| 100%/yr | $2,178 | $2,101 | +77 | SeasonalNaive |
| 157%/yr (crossover) | $2,589 | $2,589 | +0 | tie: LightGBM = SeasonalNaive |
| 730%/yr (the original 2%/day) | $6,699 | $7,471 | -772 | AutoETS |

Holding and stockout cost per fold, and the cost function they imply (`cost = holding_per_rate × daily_rate + stockout`):

| model | holding per 1.0/day of rate | stockout (rate-independent) |
| --- | --- | --- |
| AutoETS | 257,504 | 1,421.1 |
| AutoTheta | 268,084 | 1,376.9 |
| LightGBM | 261,923 | 1,460.9 |
| MovingAverage | 264,282 | 1,403.0 |
| SeasonalNaive | 311,105 | 1,249.0 |

Solving `cost_LightGBM(r) = cost_SeasonalNaive(r)`: `r* = (S_LightGBM − S_SeasonalNaive) / (H_SeasonalNaive − H_LightGBM)` = (1,460.9 − 1,249.0) / (311,105 − 261,923) = **0.004309/day = 157.27%/yr**. The two models trade one cost for the other: LightGBM carries less stock and stocks out more. Carrying less pays only when carrying is expensive; below the crossover the extra stockouts cost more than the stock saved.

## 2. Three parameters decide the ranking

The crossover is a property of the cost model *and* the service target, not of the holding rate alone:

| cost model | service target | crossover | LightGBM − SeasonalNaive at 25%/yr |
| --- | --- | --- | --- |
| unit economics (27.5% margin) — default | 95% (default) | 157%/yr | +178 [+77, +299] |
| unit economics (27.5% margin) — default | 99.9999% (cost-optimal) | 36%/yr | +44 [-61, +205] — crosses zero |
| legacy flat $5.00/unit, stock at price | 95% (default) | 364%/yr | +630 [+204, +1,089] |

## 3. Verified, not fitted

1. **Closed form** from the aggregated components: 157.2747%/yr.
2. **Independent root-find** — bisection on the re-priced *raw per-series rows*, sharing no code with (1): 157.2747%/yr (relative difference 1.5e-12).
3. **Re-simulation at 25%/yr, 27.5% gross margin** through the real simulation code from the cached forecasts (8,000 series × fold × model rows): stockouts, shipments, orders, cycles and average on-hand are **identical** to the stored run, and every priced column (holding, stockout, total) equals the stored run re-priced to within 2.3e-13. This tests both premises the analysis rests on: the policy does not react to what holding or a lost sale costs, and re-pricing — including unit economics' per-SKU price weights — reproduces a genuine re-simulation.

## 4. Sampling uncertainty on the crossover

Paired bootstrap over the 400 series (2,000 resamples, one draw shared by both models), as in `model_ci.py`, default economics, 95% target.

Crossover: **157%/yr, 95% CI [84%, 243%]**.

| holding rate | share of resamples where LightGBM is cheaper |
| --- | --- |
| 15%/yr | 0.0% |
| 20%/yr | 0.0% |
| 25%/yr | 0.0% |
| 30%/yr | 0.0% |

## 5. Against the other models

| LightGBM vs. | crossover | cheaper |
| --- | --- | --- |
| AutoETS | none | AutoETS, always |
| AutoTheta | 498%/yr | AutoTheta below it, LightGBM above |
| MovingAverage | 896%/yr | MovingAverage below it, LightGBM above |
| SeasonalNaive | 157%/yr | SeasonalNaive below it, LightGBM above |

## 6. What this depends on

- **The lost-sale margin** (27.5%, Walmart U.S. segment FY2026 gross profit rate — a lower bound on what a stockout costs). The crossover is proportional to `margin / (1 − margin)`: it would reach the top of the plausible holding range (30%/yr) at a margin of 6.7%. Full sweep: `reports/penalty_sensitivity_*.md`. Mean unit price $5.91.
- **The service target.** The 95% default was never a finding; the cost-optimal nominal target is far higher and the ranking there is different (`reports/optimal_target_*.md`).
- **The plausible range** is the project's working range. The only bound with a named source is 25%/yr (`docs/decision.md`, "Holding cost rate"); the reading of the result does not depend on the exact bounds — at the crossover a unit held for a year costs 1.6× the unit itself.
- **Track B** replaces the assumptions with measured figures; rerun `make breakeven` and the chart, the README figure and the dashboard sliders all follow.
