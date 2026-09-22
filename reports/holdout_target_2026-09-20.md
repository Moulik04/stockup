# Held-out validation of the service-level target — 2026-09-20

Target chosen on folds 1–3 only, scored on fold 4 (the most recent 28-day window; never seen by the selection). See the module docstring of `reorderpoint/holdout_target.py` for the pass criteria, which were fixed before the held-out numbers were read.

## 1. Selection on the training folds

**Matched-service rule** (smallest target at which `normal × volume_quintile` realises at least the shipped scheme's CSL at 95%, on folds 1–3): **97.0%**.

**Cost-argmin rule**, per holding-rate reading:

| rate | scheme | argmin_target |
| --- | --- | --- |
| 2%/day (legacy default) | normal × volume_quintile | 96.0% |
| 2%/month (~24%/yr) | normal × volume_quintile | 99.9% |
| 2%/year | normal × volume_quintile | 99.9% |
| 25%/yr (default) | normal × volume_quintile | 99.9% |

## 2. Held-out fold 4: `normal × volume_quintile` at 97.0% vs. the shipped scheme at 95%

Paired bootstrap over the 400 series within this one window (2,000 resamples), fine minus shipped, per model-fold row averaged over the five models:

| metric | difference (95% CI) |
|---|---|
| cost per fold, holding at 2%/day (legacy default) | $-2,038.6 [$-2,868.4, $-1,264.6] |
| cost per fold, holding at 2%/month (~24%/yr) | $-1,959.8 [$-2,841.2, $-1,152.4] |
| cost per fold, holding at 2%/year | $-1,957.3 [$-2,836.1, $-1,147.2] |
| cost per fold, holding at 25%/yr (default) | $-1,959.9 [$-2,841.0, $-1,152.5] |
| cycle service level | +2.1pp [+0.2pp, +4.0pp] |
| fill rate | +4.7pp [+2.8pp, +6.9pp] |

## 3. Is the selected target stable? (`normal × volume_quintile`, cost-argmin rule)

| rate | train_selected | held_out_optimum | regret_of_selected | stable |
| --- | --- | --- | --- | --- |
| 2%/day (legacy default) | 96.0% | 96.0% | 0.00% | yes |
| 2%/month (~24%/yr) | 99.9% | 99.9% | 0.00% | yes |
| 2%/year | 99.9% | 99.9% | 0.00% | yes |
| 25%/yr (default) | 99.9% | 99.9% | 0.00% | yes |

`regret_of_selected` is how much more the held-out fold costs at the train-selected target than at the held-out optimum. Stable = within 2 points of nominal target, or regret < 1%.

## 4. The held-out curve

Fold 4 mean cost per fold, holding at 2%/day (legacy default):

| target | normal × intermittency | normal × volume_quintile |
| --- | --- | --- |
| 0.900 | 15556.285 | 13981.618 |
| 0.920 | 15520.244 | 13785.678 |
| 0.950 | 15604.276 | 13579.559 |
| 0.960 | 15695.687 | 13551.731 |
| 0.970 | 15849.371 | 13565.645 |
| 0.980 | 16103.123 | 13663.185 |
| 0.990 | 16625.416 | 13968.326 |
| 0.995 | 17186.102 | 14384.857 |
| 0.999 | 18520.607 | 15552.217 |

Fold 4 mean cost per fold, holding at 2%/month (~24%/yr):

| target | normal × intermittency | normal × volume_quintile |
| --- | --- | --- |
| 0.900 | 9763.945 | 8922.286 |
| 0.920 | 9224.027 | 8297.838 |
| 0.950 | 8307.149 | 7232.039 |
| 0.960 | 7950.646 | 6816.016 |
| 0.970 | 7546.777 | 6347.307 |
| 0.980 | 7052.964 | 5793.932 |
| 0.990 | 6383.150 | 5059.357 |
| 0.995 | 5843.256 | 4510.083 |
| 0.999 | 4886.746 | 3649.236 |

Fold 4 mean cost per fold, holding at 2%/year:

| target | normal × intermittency | normal × volume_quintile |
| --- | --- | --- |
| 0.900 | 9580.626 | 8762.165 |
| 0.920 | 9024.761 | 8124.156 |
| 0.950 | 8076.206 | 7031.149 |
| 0.960 | 7705.526 | 6602.841 |
| 0.970 | 7284.012 | 6118.857 |
| 0.980 | 6766.539 | 5544.881 |
| 0.990 | 6058.997 | 4777.401 |
| 0.995 | 5484.271 | 4197.560 |
| 0.999 | 4455.253 | 3272.524 |

Fold 4 mean cost per fold, holding at 25%/yr (default):

| target | normal × intermittency | normal × volume_quintile |
| --- | --- | --- |
| 0.900 | 9769.417 | 8927.065 |
| 0.920 | 9229.975 | 8303.023 |
| 0.950 | 8314.043 | 7238.035 |
| 0.960 | 7957.963 | 6822.380 |
| 0.970 | 7554.621 | 6354.127 |
| 0.980 | 7061.514 | 5801.366 |
| 0.990 | 6392.826 | 5067.773 |
| 0.995 | 5853.972 | 4519.412 |
| 0.999 | 4899.626 | 3660.482 |

## 5. Walk-forward (select on earlier folds, score the next)

Holding at 2%/day (legacy default) (matched-service selection; point estimates, one 28-day window each):

| scored_fold | selected_on | selected_target | cost_diff | csl_diff_pp | fill_diff_pp |
| --- | --- | --- | --- | --- | --- |
| 2 | fold 1 | 0.970 | -1848.907 | 1.654 | 3.844 |
| 3 | folds 1–2 | 0.970 | -1612.330 | -1.541 | 2.410 |
| 4 | folds 1–3 | 0.970 | -2038.631 | 2.118 | 4.721 |

Holding at 2%/month (~24%/yr) (matched-service selection; point estimates, one 28-day window each):

| scored_fold | selected_on | selected_target | cost_diff | csl_diff_pp | fill_diff_pp |
| --- | --- | --- | --- | --- | --- |
| 2 | fold 1 | 0.970 | -1539.053 | 1.654 | 3.844 |
| 3 | folds 1–2 | 0.970 | -952.039 | -1.541 | 2.410 |
| 4 | folds 1–3 | 0.970 | -1959.842 | 2.118 | 4.721 |

Holding at 2%/year (matched-service selection; point estimates, one 28-day window each):

| scored_fold | selected_on | selected_target | cost_diff | csl_diff_pp | fill_diff_pp |
| --- | --- | --- | --- | --- | --- |
| 2 | fold 1 | 0.970 | -1529.247 | 1.654 | 3.844 |
| 3 | folds 1–2 | 0.970 | -931.141 | -1.541 | 2.410 |
| 4 | folds 1–3 | 0.970 | -1957.349 | 2.118 | 4.721 |

Holding at 25%/yr (default) (matched-service selection; point estimates, one 28-day window each):

| scored_fold | selected_on | selected_target | cost_diff | csl_diff_pp | fill_diff_pp |
| --- | --- | --- | --- | --- | --- |
| 2 | fold 1 | 0.970 | -1539.346 | 1.654 | 3.844 |
| 3 | folds 1–2 | 0.970 | -952.662 | -1.541 | 2.410 |
| 4 | folds 1–3 | 0.970 | -1959.917 | 2.118 | 4.721 |

### Decomposition on held-out fold 4: scheme vs. target

| holding rate | shipped @95% | shipped @97% | finer @95% | finer @97% | target effect (shipped) | scheme effect (@ same target) | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2%/day (legacy default) | 15604.0 | 15849.0 | 13580.0 | 13566.0 | 245.0 | -2284.0 | -2039.0 |
| 2%/month (~24%/yr) | 8307.0 | 7547.0 | 7232.0 | 6347.0 | -760.0 | -1199.0 | -1960.0 |
| 2%/year | 8076.0 | 7284.0 | 7031.0 | 6119.0 | -792.0 | -1165.0 | -1957.0 |
| 25%/yr (default) | 8314.0 | 7555.0 | 7238.0 | 6354.0 | -759.0 | -1200.0 | -1960.0 |

### Decomposition on all four folds (in-sample): scheme vs. target

| holding rate | shipped @95% | shipped @97% | finer @95% | finer @97% | target effect (shipped) | scheme effect (@ same target) | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2%/day (legacy default) | 14848.0 | 15100.0 | 13223.0 | 13219.0 | 253.0 | -1881.0 | -1629.0 |
| 2%/month (~24%/yr) | 7579.0 | 6864.0 | 7041.0 | 6237.0 | -715.0 | -627.0 | -1342.0 |
| 2%/year | 7349.0 | 6604.0 | 6845.0 | 6016.0 | -745.0 | -588.0 | -1333.0 |
| 25%/yr (default) | 7586.0 | 6872.0 | 7047.0 | 6243.0 | -714.0 | -629.0 | -1342.0 |

*The quintile-at-97% comparison bundles two changes. Where 'target effect' is most of 'total', the saving is mostly the price of carrying more stock, not a better allocation of it — and that is exactly the part that depends on the holding rate.*

## 6. Verdict against the pre-registered criteria

| holding rate | 1. held-out cost CI excludes zero (cheaper) | 2. no distinguishable CSL loss | 3. selected target stable | adopt |
| --- | --- | --- | --- | --- |
| 2%/day (legacy default) | pass | pass | pass | yes |
| 2%/month (~24%/yr) | pass | pass | pass — **vacuous: optimum at the grid edge** | yes |
| 2%/year | pass | pass | pass — **vacuous: optimum at the grid edge** | yes |
| 25%/yr (default) | pass | pass | pass — **vacuous: optimum at the grid edge** | yes |

All three criteria pass at every holding-rate reading.

**Read criterion 3 with care at 2%/month (~24%/yr), 2%/year, 25%/yr (default).** There the cost-minimising target on the training folds and on the held-out fold are both the top of the grid (99.9%), so "stable" only means both clipped the same way — the analysis cannot say where the optimum is, only that it is higher than anything tried. What *is* established at every reading is criteria 1 and 2 for the specific 97% target, and the decomposition above shows how much of that saving is the scheme and how much is simply carrying more stock.
