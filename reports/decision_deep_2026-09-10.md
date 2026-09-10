# Decision report — 2026-09-10

Reorder policy simulated cost, same folds/models as the backtest report. lead_time_days=7, service_level_target=95%, holding_cost_rate=2.0%, stockout_penalty_per_unit=$5.00.

## Summary (mean total cost per fold, across folds)

| model | holding_cost | stockout_cost | total_cost | fill_rate | turns |
| --- | --- | --- | --- | --- | --- |
| SeasonalNaive | 8582.209 | 6641.688 | 15223.897 | 0.832 | 2.274 |
| NBEATS | 5960.687 | 11248.678 | 17209.366 | 0.715 | 2.912 |

## Per-fold breakdown

| fold | model | holding_cost | stockout_cost | total_cost | fill_rate | turns |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | SeasonalNaive | 8756.450 | 6167.585 | 14924.035 | 0.838 | 2.156 |
| 1 | NBEATS | 6153.879 | 10438.338 | 16592.217 | 0.726 | 2.738 |
| 2 | SeasonalNaive | 8293.052 | 6056.228 | 14349.281 | 0.848 | 2.328 |
| 2 | NBEATS | 5047.510 | 11974.078 | 17021.588 | 0.699 | 3.223 |
| 3 | SeasonalNaive | 8705.332 | 7128.721 | 15834.053 | 0.815 | 2.233 |
| 3 | NBEATS | 6626.040 | 10525.943 | 17151.984 | 0.727 | 2.683 |
| 4 | SeasonalNaive | 8574.003 | 7214.218 | 15788.221 | 0.826 | 2.376 |
| 4 | NBEATS | 6015.320 | 12056.353 | 18071.673 | 0.709 | 3.005 |

## Phase 4 acceptance check

- Best model (**SeasonalNaive**) mean simulated cost: $15,223.90 per fold vs. SeasonalNaive's $15,223.90 (FAIL — bar is lower than SeasonalNaive).
- Savings: $0.00 per fold (0.0% of SeasonalNaive's cost).
