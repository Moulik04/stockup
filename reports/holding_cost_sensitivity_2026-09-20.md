# Holding-cost sensitivity — 2026-09-20

Every cost in the project prices holding at `holding_cost_rate` per unit-cost per **day**. That rate's intended unit was never recorded, and at the legacy default it is not a plausible carrying cost. This report re-prices the existing runs at three readings of '2%' and at the 25%/yr default, and asks which conclusions move. Method and why no resimulation is needed: see the module docstring of `reorderpoint/holding_sensitivity.py`.

## 1. Where the value came from, and what it was meant to be

- **Origin.** `HOLDING_COST_RATE=0.02` is a config default and `.env.example` line from the
  initial commit (pre-rewrite; that commit was later removed), labelled *"Track A defaults are
  illustrative, not real business numbers."* No DECISIONS entry explains it.
- **Stated unit: none.** The master prompt lists "holding cost rate, stockout penalty (all in
  config, USD)" — a rate, with no period. `docs/decision.md` documents the *mechanism* ("end-of-day
  on-hand accrues `holding_cost_rate * unit_cost * on_hand`"), which is per day, but never says what
  period the 2% was meant to describe. The report headers print it as "2.0%" bare.
- **What the code does.** `decision.simulate_series` applies it every simulated day, so at
  that value it is **~730% of unit cost per year**. `h = i·C` is conventionally annual, and this
  looks like an annual-style figure applied per day — but that is an inference from plausibility,
  not something the repo records. The intent cannot be recovered from the repo or its history.
- **Two candidate intents, very different consequences.** *2%/year* would be a unit-conversion
  slip, but 2% a year is itself well below the textbook 15–30%/yr carrying-cost range, so it is a
  lower bound rather than a best guess. *2%/month* (~24%/yr) sits inside that range. Both are run,
  alongside the legacy value, so the result does not hinge on guessing which was meant.

## 2. Does the model cost ranking move?

Mean simulated cost per fold, CSL policy, shipped scheme, same 400 series / 4 folds:

| model | 2%/day (legacy default) | 2%/month (~24%/yr) | 2%/year | 25%/yr (default) |
| --- | --- | --- | --- | --- |
| AutoETS | 14726.361 | 7859.587 | 7642.263 | 7866.074 |
| AutoTheta | 14830.457 | 7681.538 | 7455.285 | 7688.292 |
| LightGBM | 14543.379 | 7558.769 | 7337.717 | 7565.368 |
| MovingAverage | 14914.193 | 7866.680 | 7643.637 | 7873.338 |
| SeasonalNaive | 15223.897 | 6927.762 | 6665.201 | 6935.599 |

Rank (1 = cheapest):

| model | 2%/day (legacy default) | 2%/month (~24%/yr) | 2%/year | 25%/yr (default) |
| --- | --- | --- | --- | --- |
| AutoETS | 2 | 4 | 4 | 4 |
| AutoTheta | 3 | 3 | 3 | 3 |
| LightGBM | 1 | 2 | 2 | 2 |
| MovingAverage | 4 | 5 | 5 | 5 |
| SeasonalNaive | 5 | 1 | 1 | 1 |

LightGBM minus each other model, per fold, paired-bootstrap 95% CI over series (negative = LightGBM cheaper):

| other | 2%/day (legacy default) | 2%/month (~24%/yr) | 2%/year | 25%/yr (default) |
| --- | --- | --- | --- | --- |
| AutoETS | -183 [-453, +52] — crosses zero | -301 [-574, -58] | -305 [-578, -62] | -301 [-574, -58] |
| AutoTheta | -287 [-599, +15] — crosses zero | -123 [-439, +190] — crosses zero | -118 [-436, +196] — crosses zero | -123 [-439, +190] — crosses zero |
| MovingAverage | -371 [-732, -52] | -308 [-669, +9] — crosses zero | -306 [-667, +12] — crosses zero | -308 [-669, +9] — crosses zero |
| SeasonalNaive | -681 [-1,142, -183] | +631 [+205, +1,091] | +673 [+248, +1,140] | +630 [+204, +1,089] |

NBEATS (point estimates only — no per-series data to bootstrap, see README): 2%/day (legacy default) $17,209, 2%/month (~24%/yr) $11,447, 2%/year $11,265, 25%/yr (default) $11,453; SeasonalNaive at the same rates: $15,224, $6,928, $6,665, $6,936.

**The ranking is not stable.** cheapest at 2%/day (legacy default): **LightGBM**; cheapest at 2%/month (~24%/yr): **SeasonalNaive**; cheapest at 2%/year: **SeasonalNaive**; cheapest at 25%/yr (default): **SeasonalNaive**. LightGBM's cost win over SeasonalNaive does not hold at every rate — see the CI column: at the lower holding rates stockouts dominate the total, and SeasonalNaive is the model that stocks out least (its oscillating forecast is an unpriced safety margin, README Phase 7).

## 3. Does the calibration-scheme win move?

Finer scheme minus shipped scheme, both at the same 95% nominal target, mean per fold (negative = cheaper), with paired-bootstrap CI:

| cell | 2%/day (legacy default) | 2%/month (~24%/yr) | 2%/year | 25%/yr (default) |
| --- | --- | --- | --- | --- |
| volume_quintile | -1,624 [-2,191, -1,091] | -538 [-1,111, -12] | -504 [-1,083, +28] — crosses zero | -539 [-1,112, -13] |
| volume_tercile_within_intermittency | -1,389 [-1,727, -1,074] | -450 [-723, -168] | -420 [-700, -136] | -451 [-723, -169] |

The components at the rate they were simulated at (2.0000%/day (730.0%/yr)), per fold (holding / stockout):

| form | granularity | holding_cost | stockout_cost |
| --- | --- | --- | --- |
| empirical | intermittency | 6660.0 | 8002.0 |
| empirical | volume_tercile_within_intermittency | 6509.0 | 7038.0 |
| normal | intermittency | 7519.0 | 7328.0 |
| normal | volume_quintile | 6396.0 | 6828.0 |
| normal | volume_tercile_within_intermittency | 6547.0 | 6911.0 |

**The scheme win keeps its sign at every rate but not its significance or its size.** The volume-quintile scheme has lower holding ($6,396 vs $7,519) *and* lower stockout cost ($6,828 vs $7,328) at the same target, so re-weighting the two cannot reverse the direction. But the saving shrinks from $1,624 to $538 per fold at 2%/month and $504 at 2%/year, and the interval at the lower rates is at or across zero: the 11% headline saving was mostly a holding-cost saving, and holding is the term that is mispriced. The tercile scheme, by contrast, stays significant at every rate.

## 4. Does the cost-optimal service level move?

Nominal-target argmin of simulated cost, per scheme and rate (grid 80%–99.9%):

| rate | scheme | argmin | realised_csl | saving_vs_95 |
| --- | --- | --- | --- | --- |
| 2%/day (legacy default) | normal × intermittency | 92.0% | 79.5% | 0.8% |
| 2%/day (legacy default) | normal × volume_quintile | 96.0% | 81.6% | 0.2% |
| 2%/month (~24%/yr) | normal × intermittency | 99.9% (grid edge) | 93.3% | 41.1% |
| 2%/month (~24%/yr) | normal × volume_quintile | 99.9% (grid edge) | 93.4% | 48.4% |
| 2%/year | normal × intermittency | 99.9% (grid edge) | 93.3% | 45.0% |
| 2%/year | normal × volume_quintile | 99.9% (grid edge) | 93.4% | 52.1% |
| 25%/yr (default) | normal × intermittency | 99.9% (grid edge) | 93.3% | 41.0% |
| 25%/yr (default) | normal × volume_quintile | 99.9% (grid edge) | 93.4% | 48.3% |

Newsvendor critical ratio at each rate (the optimal *realised* CSL; mean price with the measured cycle length, and its spread across SKUs):

| rate | cr_mean_price | cr_p10 | cr_median | cr_p90 |
| --- | --- | --- | --- | --- |
| 2%/day (legacy default) | 0.824 | 0.683 | 0.869 | 0.975 |
| 2%/month (~24%/yr) | 0.993 | 0.985 | 0.995 | 0.999 |
| 2%/year | 0.999 | 0.999 | 1.000 | 1.000 |
| 25%/yr (default) | 0.993 | 0.984 | 0.995 | 0.999 |

Under the shipped scheme the cost-minimising target is 92.0% at 2%/day, 99.9% at 2%/month, 99.9% at 2%/year and 99.9% at the 25%/yr default. **It moves substantially with the holding rate**, so 'the optimum is ~92%' (Task 7) was a statement about the legacy 2%/day rate, not about the problem.
