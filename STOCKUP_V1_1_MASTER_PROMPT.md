# StockUp v1.1 — Master Prompt: Close the Analysis Gaps

You are the lead engineer on **StockUp** (repo name; `reorderpoint` package).
Phases 0–8 are complete and tagged `v1.0.0`. This file defines **v1.1**, a
review-driven pass that fixes what a critical reader would attack first.

The project is already good. Everything below is against a high bar. Do not
rewrite what works — `DECISIONS.md` discipline, the Phase 4 failure
narrative, the honest NBEATS negative result, and the "Where it fails"
section are the project's strongest assets and stay exactly as they are.

Read this whole file before writing code. Re-read **Hard constraints** and
**Checkpoints** before every task.

---

## Hard constraints (unchanged)

1. **Zero cost.** No paid APIs, no billed cloud, no paid tiers.
2. **No leakage, ever.** Rolling-origin only; features point-in-time safe.
   The leakage test stays mandatory for any new feature.
3. **Baselines first.** No new model reported without seasonal naive and
   moving average on the same folds.
4. **Reproducible** on an M2 Mac via `make setup`.
5. **Bridges-2** (V100 32GB, fp16 only — no bf16, no FlashAttention) for
   anything that needs a GPU. Ask before submitting jobs; hours are finite.
6. **Honest reporting.** Every task below can make a headline number look
   worse. That is an acceptable and expected outcome. A finding that
   weakens a claim gets written up, not buried — this project's existing
   Phase 4 and Phase 7 entries are the standard to match.

---

## Task 1 — Are the models actually different? (do this first, it gates the rest)

RMSSE is 0.814–0.823 for AutoTheta, MovingAverage, AutoETS and LightGBM —
a suspiciously tight cluster on a 77%-zero panel, where predicting
approximately flat near-zero minimises squared error for everyone.

Before any further modelling work, measure whether the model comparison is
measuring anything:

- Pairwise Pearson and Spearman correlation between models' P50 forecast
  vectors, pooled and per intermittency bucket.
- Mean absolute difference between model forecasts, as a fraction of mean
  demand.
- Fraction of series where any two models' 28-day forecasts differ by less
  than 5%.
- Plot: five models' forecasts for a handful of representative series
  (one high-volume, one intermittent, one cold-start).

Write `reports/model_divergence_<date>.md`.

**If the models are >0.95 correlated**, that is the single most important
finding in the project and it reframes everything: the accuracy and cost
tables are ranking noise, and the README must say so. **If they diverge
meaningfully**, the comparisons stand and you have evidence they do.

Either way this belongs in the README. Do not skip it because the answer
might be inconvenient.

## Task 2 — Fill rate vs cycle service level: the metric mismatch

`docs/decision.md` sizes safety stock as `z = Φ⁻¹(service_level_target)`
with `service_level_target = 0.95`. That is a **cycle service level** — the
probability of no stockout during a replenishment cycle. The cost
simulation reports **fill rate** — the fraction of demand met from stock.
These are different quantities and a 95% CSL does not imply a 95% fill
rate.

This compounds with decision #3 (`S = reorder_point`, no separate lot
size). Fill rate is approximately `1 − σ_LT·G(z)/Q`, where G is the
standardised loss function and Q the order quantity. With S = s, Q is only
the deficit consumed since the last order — small. Small Q mechanically
depresses fill rate relative to CSL, and on intermittent demand where
σ_LT is comparable to μ_LT, an 80–83% fill rate at nominal 95% CSL is
close to what the formula predicts.

Do this:

1. **Measure realised cycle service level** — fraction of replenishment
   cycles with no stockout — alongside fill rate, for every model, in the
   existing cost simulation. Report both in `reports/decision_<date>.md`
   and the README.
2. **Compare.** If CSL lands near 95% while fill rate sits near 81%, the
   "every model misses the service-level target" conclusion in the README
   and in the 2026-09-10 DECISIONS entry is measuring the wrong thing and
   both must be rewritten. If CSL also comes in near 81%, the original
   under-provisioning read was right and is now properly evidenced.
3. **Add a fill-rate-targeted policy** as a second option alongside the
   CSL policy: size safety stock to hit a target fill rate directly, via
   the loss-function relationship rather than `z·σ`. Report both policies'
   cost and service outcomes. Keep the CSL policy — the comparison is the
   point.

**Acceptance:** both service measures reported for every model; the README
and DECISIONS entry reflect whichever conclusion the data supports; a test
covers the CSL computation.

## Task 3 — Confidence intervals on every comparison

Four folds and 400 series, with every result reported as a point estimate.
LightGBM beats AutoETS by 1.2% on cost; MinTrace improves MASE by 0.10%.
Nothing in the repo says whether those are distinguishable from zero.

Implement a **paired bootstrap over series**: resample series with
replacement, recompute each model's metric on the resample, and report the
distribution of the *difference between models* (paired, so fold and series
effects cancel). 2,000 resamples is plenty.

Report, for every headline comparison in the README:

- point estimate of the difference,
- 95% bootstrap CI on the difference,
- fraction of resamples where the sign flips.

Apply to: LightGBM vs SeasonalNaive on cost (the headline claim), LightGBM
vs each statistical baseline, MinTrace vs base at each hierarchy level, and
NBEATS vs LightGBM on both accuracy and cost.

Expect some claims not to survive. The MinTrace result (−0.10%) very
likely will not, and the LightGBM-vs-statistical-baselines cost gap may
not either. **That is the point.** A claim that survives a bootstrap is
worth ten that do not, and "we tested our own rankings for significance
and two of them did not hold" is a stronger portfolio line than an
unqualified table.

**Acceptance:** every README results table carries CIs on the differences;
any claim whose CI crosses zero is restated as "not distinguishable at this
sample size" rather than as a ranking.

## Task 4 — Add the Croston family

Intermittent demand is this project's stated central difficulty — first
bullet in "Where it fails", the justification for the Tweedie objective,
the reason for two-bucket residual pooling. But the modelling ladder never
includes the models designed for it.

Add a Croston rung to the registry: `CrostonClassic`, `CrostonOptimized`,
`CrostonSBA`, and `TSB` (all in `statsforecast`, already a dependency).
Consider `ADIDA`/`IMAPA` if cheap. Run them through the identical
backtest and decision-cost path as every other model.

Two implementation notes:

- Croston-family models produce a point forecast without native prediction
  intervals. Use the same conformal-interval approach
  `StatsForecastQuantileModel` already applies to the other statsforecast
  models, and apply the non-negativity clip from code-review finding #2.
- Report results split by intermittency bucket, not just pooled. Croston's
  advantage, if any, should concentrate in the >50%-zero bucket, and
  pooling will dilute it.

They may lose. A Croston rung that loses, reported honestly, closes the
most obvious gap in the modelling ladder. Its absence does not.

**Acceptance:** Croston family in the registry, in the backtest table, in
the decision-cost table, with a per-bucket breakdown; README ladder updated.

## Task 5 — Stratify the residual pooling, and diagnose it

The 2026-09-05 empirical fix pools residuals across ~400 series into two
intermittency buckets. Every series in a bucket then receives the same
absolute safety-stock σ — so a high-volume SKU and a low-volume SKU are
provisioned identically in units. On a panel with wide volume dispersion
that under-provisions the high-volume tail and over-provisions the low
end, which would depress aggregate fill rate exactly as observed.

Diagnose before fixing:

1. Plot realised fill rate against series volume decile, per model. A
   declining relationship confirms the pooling hypothesis and is a second
   candidate explanation for the service-level gap, independent of Task 2's
   metric mismatch.
2. Report the within-bucket dispersion of per-series residual std. If it
   spans an order of magnitude, pooling in absolute units is indefensible.

Then, only if the diagnosis supports it, add a volume dimension to the
bucketing (intermittency × volume tercile), or pool a *relative* quantity
(coefficient of variation) and rescale per series. Keep the current scheme
as a reported ablation row either way.

**Acceptance:** the fill-rate-vs-volume plot in the report; a decision
recorded in `DECISIONS.md` with the dispersion numbers behind it.

## Task 6 — Confront the MASE ranking in the README

LightGBM is 4th of 5 on MASE, behind a moving average. This is disclosed
but framed defensively ("beats seasonal naive in 4/4 folds", "only 4%
worse"). A reader who looks at the table sees the ranking immediately, and
the framing costs more credibility than the result itself does.

Rewrite the Results section to state it plainly and then make the actual
argument, which is a good one:

- LightGBM ranks 4th of 5 on MASE, behind the statistical baselines.
- This is a known and expected result on intermittent demand, where global
  ML models frequently lose to simple per-series methods on point accuracy.
- LightGBM is nonetheless the production model because it is the only one
  clearing the quantile-coverage bar, and because it wins on simulated
  decision cost — which is the metric this project argues should decide.
- That is the project's thesis working, not an embarrassment: a model can
  lose on RMSE-style accuracy and still be the right choice.

Also drop or rename the simplified WRMSSE. It is a single-level
revenue-weighted RMSSE documented as not being WRMSSE, so the name only
invites a false comparison to the M5 leaderboard. Call it
`revenue_weighted_rmsse` and say in one line what it is.

**Acceptance:** README states the ranking before defending it; the metric
is renamed; no claim in the README overstates what the numbers support.

## Task 7 — Cost-optimal service level

`service_level_target = 0.95` is arbitrary, and decision #2 notes it sizes
safety stock directly rather than from the stockout/holding ratio a real
newsvendor optimum uses. But the cost simulation already prices both sides.

Close the loop: compute the critical ratio `Cu / (Cu + Co)` from the
project's own `stockout_penalty_per_unit` and `holding_cost_rate`, and
report the cost-optimal service level implied by the configured costs. Then
run the cost simulation across a sweep of service-level targets (0.80 to
0.99) and show where simulated cost actually minimises.

This is a small change that directly serves the project's thesis: it turns
"we picked 95%" into "95% is not cost-optimal for these cost parameters;
the simulation minimises at X%, and here is the curve." It also gives the
Streamlit dashboard a genuinely useful control.

**Acceptance:** cost-vs-service-level curve in the decision report; the
implied optimum stated; README notes whether 95% was a good choice.

## Task 8 — Track B

The master prompt's differentiator was always Track B: the same pipeline
pointed at MJ's family business data, so the project is MJ's rather than
everyone's M5 project. The loader is built and fixture-tested; the export
has not been provided.

This is the highest-value remaining work for portfolio purposes and it is
blocked on data, not code. When the export arrives:

- Run the identical pipeline via the config switch. Change nothing to make
  it fit.
- Report **aggregated results only** — no SKU names, no absolute revenue,
  no customer information. Percentages, index numbers, and relative
  improvements are sufficient to tell the story.
- Expect monthly rather than daily grain, far fewer series, and strong
  annual seasonality. The fold structure and horizon will need config
  changes; the code should not.
- Write the case study as a short README section: what the data is at a
  level of abstraction that gives nothing away, what the pipeline
  recommended, and how that compares to current practice.

**Checkpoint:** confirm with MJ what can be published before writing
anything into the repo.

## Task 9 — Demote Phase 5 in the README

The reconciliation experiment used AutoETS as its base because LightGBM's
price/calendar features have no definition at synthetic nodes — a correct
call, but it means the experiment says nothing about the production model,
and the effect sizes (−0.10% to −0.12%) are almost certainly inside the
noise band Task 3 will now compute.

Keep the work and the report — a negative result on reconciliation is
legitimate. But move it out of the main Results flow into a shorter
subsection, state that it was run on a non-production base model, and
attach Task 3's CIs. If the CI crosses zero, say so directly.

---

## Definition of done for v1.1

1. Model divergence measured and reported; if models are near-identical,
   the README leads with that.
2. Both cycle service level and fill rate reported; the service-level
   conclusion matches whichever the evidence supports.
3. Every headline comparison carries a bootstrap CI on the difference; any
   claim crossing zero restated as inconclusive.
4. Croston family evaluated, per intermittency bucket.
5. Residual-pooling diagnosis done; fill-rate-vs-volume plot published.
6. README states the MASE ranking before defending it; WRMSSE renamed.
7. Cost-optimal service level computed and compared against the configured
   0.95.
8. Test suite still green; no new leakage-test gaps.

## Checkpoints — stop and ask MJ

- Before any Bridges-2 job.
- Before publishing anything derived from Track B data.
- Before removing any existing report — the historical-snapshot convention
  (`decision_2026-09-03.md` alongside `decision_2026-09-05.md`) stays.
- If Task 1 shows the models are near-identical, or Task 3 invalidates a
  headline claim: stop and report before rewriting the README. Those
  findings change what the project's story is, and MJ should decide how to
  tell it.

At the end of each task: what was measured, what changed, which README or
DECISIONS claims it affects, and one question for MJ.
