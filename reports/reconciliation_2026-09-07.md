# Hierarchical reconciliation report — 2026-09-07

Track A (M5), stores CA_1, TX_1 (full state/store/category/department hierarchy under them), 400-series bottom-level subsample (seed=0). 4 rolling-origin folds, horizon=28 days, base model=AutoETS. Bottom-up and MinTrace (shrinkage) via `hierarchicalforecast` — see docs/decision.md's sibling design note in DECISIONS.md for the simplified (nested, not grouped) hierarchy this uses.

## Mean MASE by level and reconciliation method

| level | BottomUp | MinTrace | base | BottomUp_vs_base | MinTrace_vs_base |
| --- | --- | --- | --- | --- | --- |
| state_id | 0.774 | 0.755 | 0.747 | 3.641 | 1.066 |
| state_id/store_id | 0.774 | 0.755 | 0.747 | 3.641 | 1.066 |
| state_id/store_id/cat_id | 0.804 | 0.794 | 0.794 | 1.261 | 0.064 |
| state_id/store_id/cat_id/dept_id | 0.856 | 0.848 | 0.849 | 0.904 | -0.099 |
| state_id/store_id/cat_id/dept_id/item_id | 1.329 | 1.327 | 1.329 | 0.000 | -0.122 |

n_nodes per level (mean across folds): state_id=2, state_id/store_id=2, state_id/store_id/cat_id=6, state_id/store_id/cat_id/dept_id=14, state_id/store_id/cat_id/dept_id/item_id=400

## Does reconciliation help, and where?

Reconciliation (BottomUp and/or MinTrace) improves mean MASE over the unreconciled base forecast at: state_id/store_id/cat_id/dept_id, state_id/store_id/cat_id/dept_id/item_id. It makes things worse at: state_id, state_id/store_id, state_id/store_id/cat_id. The bottom level is expected to be least affected by BottomUp specifically (it's definitionally a no-op there — nothing to sum up from below); MinTrace can still adjust bottom-level forecasts using the whole hierarchy's residual covariance.

Bottom-level base MASE (1.33) is 1.8x top-level base MASE (0.75) — bottom-level (item x store) series are simply noisier/harder to forecast than an aggregate. BottomUp reconciliation just sums these noisier bottom forecasts, so it inherits that noise at every level above the bottom; MinTrace's covariance-weighted blend is specifically designed to correct for this rather than trusting the bottom level uniformly, which is consistent with it degrading less (or even improving on) the base forecast where BottomUp doesn't.
