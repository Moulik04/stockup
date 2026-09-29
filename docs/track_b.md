# Track B — UCI Online Retail II: design, registered before any model was run

Written and committed on 2026-09-29, when the data had been cleaned and described (`reports/
track_b_cleaning_*.md`) and **no model had been run on it**. The commit history dates this file. Anything
that changes after a result is seen goes in "Changes after registration" at the bottom, with the reason,
rather than into the text above it.

Data: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CG6D>, CC BY 4.0. A UK online seller of all-occasion gift-ware, many of
whose customers are wholesalers; 2009-12-01 to 2011-12-09. Cleaning rules: `docs/data.md`.

## What Track B is for

External validity. Track A is M5, a giant US retailer selling FOODS, HOUSEHOLD and HOBBIES daily
across stores. Track B is a small UK online seller with wholesale customers, ordering in bulk, weekly.
The question is whether StockUp's findings replicate at a very different real business. Replication and
non-replication are reported with equal weight, and what counts as each is fixed below, before the
results.

## The findings under test, and what "replicated" will mean

Each rule is set in advance so the answer cannot be adjusted to the result. **Every verdict is judged on the
primary panel (below).** The robustness panel is reported alongside as agrees or disagrees, under the same
rule, and is never used to choose the more favourable result. All use paired bootstrap
95% intervals (series resampled, 2,000 draws, as elsewhere in the repository) and the shipped cell
(`S = s + 2 × lead-time demand`, 95% target, normal form, intermittency buckets) unless stated.

1. **The ordering policy matters more than the model.** Track A: the policy change was worth about
   71% of pooled cost; the spread between models under the repaired policy was a small fraction of
   that. *Replicated* if, for every model, moving from `S = s` to `S = s + 2 × lead-time demand`
   lowers pooled cost with a CI that excludes zero, **and** the mean of that saving across models is
   larger than the spread between the cheapest and dearest non-naive model under the shipped policy.
   Otherwise not.
2. **The calibrated quantile does what it says; the shortfall is the policy.** Track A: the reorder
   point covered held-out lead-time demand in 94.8-95.1% of windows against a 95% target, while
   realised cycle service level under `S > s` was 91.9%, with about half of the remaining stockouts
   undershoot. *Replicated* if held-out coverage of the reorder point is within 3 points of the
   target **and** realised CSL under `S > s` is at least 3 points below that coverage **and** at
   least a third of the remaining stockout cycles are undershoot. If coverage misses by more than 3
   points the calibration itself did not transfer, which is a different and more serious result.
3. **The model cluster is a tie on cost.** Track A: MovingAverage, AutoETS, AutoTheta, LightGBM and
   the Croston family were all cost-tied. *Replicated* if every pairwise paired-bootstrap CI among
   those models, pooled over series and folds, contains zero. Any model whose CI excludes zero is
   named as a non-replication with its interval.
4. **The trailing mean is enough.** The v1.2 production model, a trailing mean, cost the same as
   LightGBM (+$11 per fold, CI containing zero). Track B's analogue is a 4-week mean. *Replicated* if
   its CI against LightGBM contains zero.
5. **Reported either way, no pre-registered threshold:** the model accuracy ranking (MASE), whether
   accuracy and cost rank models the same, the effect of the lead time and the margin (below), and
   how the bulk orders show up in the residuals and in the stockouts.

### How each rule will be read, fixed before any model was run

The rules above leave a few choices open. They are closed here, in advance, so that no verdict depends
on a reading picked after seeing a number. "The cluster" is the seven non-naive models: MovingAverage,
AutoETS, AutoTheta, LightGBM, CrostonClassic, CrostonSBA, TSB. SeasonalNaive is the eighth model and is
in the ladder, the tables and finding 1's "every model", but not in the cluster.

- **The default cell** is margin 27.5%, lead time 2 weeks, 95% target. Verdicts are taken there. The other
  eleven cells of the 4 × 3 sensitivity grid are computed the same way and reported as a grid, and the
  headline says in how many of the twelve the default verdict holds.
- **Finding 1.** "Every model" is all eight. The saving is the mean over the eight models of cost per fold
  at `S = s` minus cost per fold at `S = s + 2 × lead-time demand`. The spread is the highest minus the
  lowest cost per fold among the seven cluster models under the shipped policy. Replicated if all eight
  savings have 95% CIs that exclude zero and the saving exceeds the spread.
- **Finding 2.** "The reorder point" is every cluster model: each must have held-out coverage within 3
  points of 95%. Coverage is measured on fixed-origin windows of the lead time's length inside the test
  folds, with the calibration from the window before fold 1, exactly as in `reports/production_model_*.md`
  §4. The coverage figure used for the gap is the mean over the cluster; realised CSL and the undershoot
  share are pooled over the cluster's cycles under `S > s`. The three conditions are as registered.
- **Finding 3.** Read literally: all 21 pairs among the seven cluster models must have a CI containing
  zero. With 21 comparisons even seven truly tied models would show about one CI excluding zero by
  chance (5% of 21, less with the correlation between pairs), so a lone marginal exclusion is a weak
  reason to call the cluster untied; the verdict is still mechanical, the count and intervals are
  reported, and the six comparisons against LightGBM (Track A's design) are reported beside the 21.
- **Finding 4.** The 4-week mean (the ladder's MovingAverage) against LightGBM, pooled over folds and
  series, CI containing zero.
- **The added stratum.** The 924 SKUs the robustness panel adds are broken out for the cost table and
  the accuracy table only, descriptively.

### Known consequences of the design, stated before the results

- **The calibration is one window.** As in Track A it is the residuals of the lead-time window before
  fold 1, here the last 2 weeks of the first 52, which are late November 2010: the autumn peak. That one
  window sizes the buffer for all four folds, so a buffer set at peak volume is applied in summer. If the
  buffer is too large in the quiet folds, that is a property of a single calibration window, and it is
  the same design choice Track A made with a single, unseasonal window.
- **SeasonalNaive on the calibration window.** It needs 52 weeks and that window has 50, so it falls back
  to the plain naive forecast there (the fold forecasts use the 52-week season: fold 1 has exactly 52).
  AutoETS and AutoTheta are non-seasonal in every fold (no window holds two annual cycles).

## Economics, fixed in advance

Track A's economics are documented assumptions with sensitivity. Track B's are the same kind of thing:
the public data has no costs, margins or lead times.

- **Currency.** Prices are pounds sterling. Every cost is reported in **USD at one fixed rate,
  1.5770 USD per GBP**: the mean of the 25 monthly averages, December 2009 to December 2011, of the
  Federal Reserve Board's H.10 series "U.S. dollars to one British pound", as published in FRED as
  `EXUSUK` (<https://fred.stlouisfed.org/series/EXUSUK>). One rate, not the rate on the day of sale,
  so no result depends on when a sale happened. (The monthly figures ranged from 1.4669 to 1.6379.)
- **Unit value.** Observed: the weekly median line price of the SKU, in USD, carried forward over
  weeks with no sale. Lines priced more than 10× or under 1/10 of the SKU's median are excluded from
  that price and never from demand.
- **Gross margin: 27.5%, an assumption with no source specific to this business.** It is Track A's
  figure (Walmart U.S., fiscal 2026 10-K), held equal on purpose so that any difference between the
  tracks is about the business and the grain, not about the economics. It is very probably too low
  for a wholesaler of gift-ware, which understates what a lost sale costs. Sensitivity is therefore
  part of the design, not an afterthought: **20%, 27.5%, 40% and 50%**, all four reported, and the
  headline states which findings survive across the grid.
- **Holding cost: 25% of unit value per year**, Track A's assumption (a cross-industry average, not a
  measurement for this business), applied weekly as 25% / 52.
- **Lead time: 2 weeks**, an assumption: supplier lead times are not in the data, and a UK seller of
  partly imported gift-ware plausibly sits between one and several weeks. Sensitivity: **1 and 4
  weeks**. With weekly periods a 1-week lead time is a single forecast step, so the lead-time total is
  no longer an aggregation of a horizon, which is a difference from Track A worth stating in the
  result.
- **Service target 95%, lot multiple 2.0** (Track A's shipped cell), plus `S = s` (lot multiple 0) for
  finding 1. No ordering cost is priced, as in Track A.

## Series selection: a primary panel and a pre-declared robustness panel

**Primary panel: a SKU is in if it sold in at least 26 of the first 52 weeks (1,742 SKUs).** Fixed before
any model was run, and unchanged.

**Robustness panel: at least 13 of the first 52 weeks (2,666 SKUs).** Added on 2026-09-29 after the
descriptive cleaning report and before any model was run (see "Changes after registration": an addition,
not a change). It is a strict superset of the primary panel, 924 SKUs larger. Its rule is registered here,
in advance, exactly as the primary's is:

- **The verdict on each finding is the primary panel's.** The robustness panel gets the identical rule,
  applied to its own results, and is reported as **agrees** (same verdict) or **disagrees** (different
  verdict) next to the primary's. The headline states the primary verdict first; where the panels
  disagree, the headline says so. Neither panel's result replaces the other, and the robustness panel is
  never used to pick the more favourable of two answers.
- Everything else (economics, folds, models, sensitivity grid, bootstrap) is identical across panels.
- Also reported, descriptively and with no verdict attached: the 924 SKUs the robustness panel adds, as a
  separate stratum, because they are the short-season end of the catalogue (below).

### What the two panels cover, and what they do not

- **Both select on first-year activity, which is correct**: it looks at nothing in the test folds, so there
  is no look-ahead. The same property means **both exclude the 652 SKUs first sold in year 2** (19.3% of
  year-2 units). The findings apply to products with at least a year of history. Cold start, a product with
  no history to forecast from, is a separate known limitation that neither panel addresses.
- **The primary panel is declining**: its year-2 units are 73% of year 1, against 94% company-wide.
  That is the pattern selecting strong year-1 sellers produces, since they then fade. It tests **mature
  products with muted seasonality** (Sep-Nov over Mar-Jul: 1.34 in year 1, 1.28 in year 2).
- **The robustness panel is where more of the seasonal products are, but it is not a substitute for
  them.** Measured before any model: its Sep-Nov over Mar-Jul is 1.47 (year 1) and 1.40 (year 2); the 924
  SKUs it adds run 2.81 and 2.32 on their own, but carry only 11.1% of company units. Mean weekly units in
  fold 4's window (2011-09-05 to 2011-11-28) against the 39 weeks before it: **1.32 for the primary panel,
  1.43 for the robustness panel, 1.81 for the company.** The robustness panel also declines, to the same
  73% of year 1. So the autumn-peak test lives mainly in the robustness panel, and mainly in its added
  stratum, but neither panel reproduces the company's build-up, and no result here says how the models
  cope with the company's peak as a whole.
- Weekly zero share (open weeks): primary 28.8%, robustness 41.7%, the added stratum 66.1%, all cleaned
  SKUs 59.9%.

## Evaluation design

- **Grain and length.** 104 complete Monday-to-Sunday weeks, 2009-12-07 to 2011-11-28. Weekly, because
  two years is too short for monthly folds with annual seasonality.
- **Folds.** Rolling origin, expanding window, **4 folds of a 13-week horizon** covering the final 52
  weeks as test, gap 0. Fold 4's test window is 2011-09-05 to 2011-11-28, the autumn build-up, so the
  final fold includes the 2011 holiday peak; fold 1 starts 2010-12-06 and its window contains the
  Christmas closure week (2010-12-27), which no model can be expected to anticipate from one prior
  year. **The limitation to state with every result: 2010 is the only prior year.** A model can see
  the previous holiday season once, in the training data of folds 3 and 4, and no model can learn the
  annual pattern from a single cycle beyond repeating it.
- **Models.** The same ladder as Track A through the same registry: SeasonalNaive (season length 52),
  MovingAverage (4-week mean), AutoETS, AutoTheta, the Croston family, and LightGBM with weekly
  analogues of the daily features (recent lags and rolling means, week of year). No tuning on the test
  folds. A model that cannot run on the available history (a season of 52 needs two full cycles) falls
  back to its non-seasonal form, and the fallback is reported.
- **Sample.** All selected series. If a model takes more than an hour on that, every model is run on
  the same fixed random subset of 400 series (seed 0), as in Track A, and the change is recorded below.
- **Decision layer.** The calibration uses the residuals of the lead-time window immediately before
  fold 1, as in Track A's harness, with the intermittency buckets. Bootstrap intervals as elsewhere.
- **Not changed by the grain:** `y` is units invoiced, a demand proxy (a week with no sale may be a
  week with no stock); bulk orders are kept as demand; cancellations are netted as in `docs/data.md`.

## Known limits, stated before the results

- One business, two years, one prior holiday season.
- Series selection is by year-1 activity (above): products with at least a year of history only, cold
  start not covered, and both panels decline over the test period.
- `y` is sales, not demand; stockouts are unobserved.
- Costs, margin and lead time are assumptions; the sensitivity grid is the defence.
- Weekly aggregation removes most zero periods (the panel is about 29% zeros against Track A's 77%),
  but leaves a lumpier non-zero week driven by wholesale orders.
- Comparing tracks confounds business type with grain (daily against weekly) and with catalogue
  (steady sellers against everything): a difference is not evidence about business type alone.

## Changes after registration

- **2026-09-29, before any model was run: robustness panel added** (at least 13 of the first 52 weeks,
  2,666 SKUs), alongside the unchanged primary panel (at least 26, 1,742 SKUs). Reason: the cleaning
  report showed, from descriptive statistics alone, that the registered rule keeps year-round sellers and
  understates the seasonal end of the catalogue. Decided by the project owner from that report. This is an
  addition: the primary rule, the economics, the folds and the replication rules are untouched, and the
  robustness panel's role (agrees or disagrees, never a substitute) was registered in the same commit that
  introduced it.
