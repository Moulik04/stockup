# Reconciliation confidence intervals — 2026-09-18

Paired bootstrap over hierarchy nodes (2,000 resamples, seed 0) on mean MASE, MinTrace and BottomUp vs the unreconciled AutoETS base, per level. Base model is AutoETS (not the production LightGBM — its price/calendar features have no definition at a synthetic aggregate node), so this experiment says nothing about the production model regardless of what the intervals show.

| level | method | n_nodes | base_mase | mase_diff | diff_pct | ci_pct | sign_flips |
| --- | --- | --- | --- | --- | --- | --- | --- |
| state_id | MinTrace | 2 | 0.747 | 0.008 | +1.07% | not testable (too few nodes) | — |
| state_id | BottomUp | 2 | 0.747 | 0.027 | +3.64% | not testable (too few nodes) | — |
| state_id/store_id | MinTrace | 2 | 0.747 | 0.008 | +1.07% | not testable (too few nodes) | — |
| state_id/store_id | BottomUp | 2 | 0.747 | 0.027 | +3.64% | not testable (too few nodes) | — |
| state_id/store_id/cat_id | MinTrace | 6 | 0.794 | 0.001 | +0.06% | not testable (too few nodes) | — |
| state_id/store_id/cat_id | BottomUp | 6 | 0.794 | 0.010 | +1.26% | not testable (too few nodes) | — |
| state_id/store_id/cat_id/dept_id | MinTrace | 14 | 0.849 | -0.001 | -0.10% | not testable (too few nodes) | — |
| state_id/store_id/cat_id/dept_id | BottomUp | 14 | 0.849 | 0.008 | +0.90% | not testable (too few nodes) | — |
| state_id/store_id/cat_id/dept_id/item_id | MinTrace | 400 | 1.329 | -0.002 | -0.12% | [-0.24%, +0.07%] — crosses zero | 0.093 |
| state_id/store_id/cat_id/dept_id/item_id | BottomUp | 400 | 1.329 | 0.000 | +0.00% | identical to base by construction (BottomUp is a no-op at the bottom level) | — |

Levels with fewer than 30 nodes are not bootstrapped: resampling 2 states or 6 categories with replacement is not inference, and an interval there would look more informative than it is. Those rows are point estimates only, and the README's claims about them (BottomUp worse at every aggregate level; MinTrace roughly neutral above the item level) cannot be tested by this method.

At the bottom level (item × store, 400 nodes) MinTrace's MASE change is -0.0016 (95% CI -0.0032 to +0.0010) — **not distinguishable from zero.** The README's `-0.12%` is inside the noise and should be reported as such, not as a real improvement.
