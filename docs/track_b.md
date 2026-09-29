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

Each rule is set in advance so the answer cannot be adjusted to the result. All use paired bootstrap
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

## Series selection

A SKU is in if it sold in at least **26 of the first 52 weeks**. This was fixed before any model was
run. The cleaning report then showed, from descriptive statistics alone, what it does: it keeps
year-round sellers and cannot select a SKU first sold in year 2 or a short-season one, so the panel
shrinks to 73% of its year-1 units in year 2 while the company as a whole is flat (94%), and its
autumn peak is much weaker than the company's (`reports/track_b_cleaning_*.md`). That is a property of
the rule, and the results will be read as **results for the catalogue's steady sellers**, not for the
business as a whole. Whether to add a second panel (for example a rule that admits seasonal SKUs) is
decided with that report in hand and recorded under "Changes after registration" if it happens.

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
- Series selection is by year-1 survival (above).
- `y` is sales, not demand; stockouts are unobserved.
- Costs, margin and lead time are assumptions; the sensitivity grid is the defence.
- Weekly aggregation removes most zero periods (the panel is about 29% zeros against Track A's 77%),
  but leaves a lumpier non-zero week driven by wholesale orders.
- Comparing tracks confounds business type with grain (daily against weekly) and with catalogue
  (steady sellers against everything): a difference is not evidence about business type alone.

## Changes after registration

None yet.
