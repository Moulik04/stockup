# Decision layer

Converts a P10/P50/P90 quantile forecast into a per-SKU reorder point, safety stock, and order
quantity, then simulates the $ cost those decisions would have produced against realised demand —
the headline number this project compares across models.

## Newsvendor-style service-level policy

Inputs per series: the forecast's `p10`/`p50`/`p90` for each of the next `lead_time_days`, plus
`on_hand` (current stock) and `service_level_target` (e.g. 0.95) from `CostParams`.

**1. Lead-time demand distribution.** P10/P90 is this project's central 80% interval (see
`models/base.py`), so under a normal approximation each day's demand has

```
daily_std = (p90 - p10) / (2 * Z80),   Z80 = Φ⁻¹(0.9) ≈ 1.2816
```

Summed over the lead time, assuming day-to-day forecast errors are independent:

```
lead_time_mean = Σ p50_i                for i in the next lead_time_days forecast rows
lead_time_std  = sqrt(Σ daily_std_i²)
```

**2. Safety stock and reorder point.**

```
z = Φ⁻¹(service_level_target)
safety_stock = z * lead_time_std
reorder_point (s) = lead_time_mean + safety_stock
```

`lead_time_mean` always comes from the forecast (above). `lead_time_std` has two sources,
depending on whether backtest history exists yet:

- **No history (cold start / a single live forecast call, e.g. a future serving API):**
  `lead_time_std` from step 1 — the forecast's own quantile spread.
- **Backtest evaluation (this project's actual `make decide` path):** an **empirical, pooled
  residual std** from the model's own realised forecast errors on earlier folds — see "Empirical
  safety-stock sizing" below. This is the one that matters for every number in
  `reports/decision_<date>.md`.

Trusting each model's self-reported P10/P90 width for sizing turned out to be a real bug in
practice, not just a theoretical concern. Two models
can be equally (or even differently, in either direction) well-calibrated on *marginal* coverage
while producing very different *absolute* interval widths, because that width is a property of
how each model family fits (independently-trained quantile regressors vs. conformal calibration),
not a normalized, comparable "uncertainty" unit. A model doesn't have to be wrong to lose under
this formula if raw quantile width is what sizes the safety stock — it only has to be narrower.

## Empirical safety-stock sizing (backtest path)

Instead of trusting a model's stated interval, measure how far off its **P50 forecast actually
was**, historically, over a `lead_time_days` window, and use that empirical spread:

```
residual = actual_lead_time_demand - predicted_lead_time_demand     (lead_time_forecast_residuals)
```

A single past fold gives only one residual per series — nowhere near enough to estimate a
per-series std (this project only has 4 folds total). Instead, residuals are **pooled across all
~400 series** in a fold, split into the same two intermittency buckets (`zero_rate > 50%` vs.
`<= 50%`) `backtest.py`'s own "where the model fails" analysis already uses
(`empirical_lead_time_std`). Pooling trades per-series precision for a sample size (~200 points
per bucket) actually large enough to trust, rather than a per-series std computed from 1–3 folds
of history.

**Chaining across folds, without leakage:** fold *i*'s safety stock uses the *previous* fold's
own realised residuals (that forecast and its outcome are already known and fully in the past by
the time fold *i* is scored) — never fold *i*'s own future test data. Fold 1 has no previous fold,
so it gets a one-off bootstrap calibration instead: fit the model on everything except the last
`lead_time_days` of its own training window, forecast that held-back stretch, and measure
residuals there (`_bootstrap_lead_time_std` — one extra fit per model, done once, not per fold).

**3. Order quantity.** The order-up-to level is `S = s + Q`, with `Q = lot_multiple x` expected
lead-time demand (`LOT_MULTIPLE`, default 2.0 — see "Order-up-to level" below). Before 2026-09-20
this project used `S = s`, and every historical report was produced that way (they pin
`lot_multiple = 0` explicitly). There is still no EOQ: Track A has no ordering cost. When
`on_hand < s`:

```
order_quantity = max(0, S - on_hand)
```

**4. Stockout probability.** Probability that lead-time demand exceeds current on-hand, under the
same normal approximation:

```
stockout_probability = 1 - Φ((on_hand - lead_time_mean) / lead_time_std)
```

**5. Rationale.** One line per SKU: on-hand vs. expected lead-time demand, the resulting order
recommendation, and the target service level — e.g. *"On-hand (12) covers 1.8 days of expected
demand (6.8/day); lead time is 7 days at 95% service level → order 42 units."*

Note: `service_level_target`, not the stockout/holding cost ratio, sizes the safety stock — a
simple, explainable policy chosen deliberately over a full newsvendor cost-ratio optimum.
`holding_cost_rate` and
`stockout_penalty_per_unit` do not affect *how much* to order; they only price the simulation
below, which is how different service levels or models get compared in dollar terms.

### Two service measures, and why they differ (v1.1)

`service_level_target = 0.95` above is a **cycle service level (CSL)** — the probability of
surviving a replenishment cycle without a stockout. The cost simulation's headline service number
has always been **fill rate** — the fraction of demand met from stock. These are different
quantities, and a 95% CSL does not imply a 95% fill rate:

```
fill_rate ≈ 1 − σ_LT · G(k) / Q,     G(k) = φ(k) − k·(1 − Φ(k))     (standard_loss)
```

`Q` is the demand consumed per replenishment cycle. Under the original `S = s` (step 3 below;
the fill-rate reports were produced that way), `Q` is only the deficit accumulated since the last order — *small*. Fill rate depends on
the shortfall per cycle relative to `Q`, so a small `Q` mechanically drives fill rate away from
CSL; the sizing rule `z · σ_LT` never targeted fill rate in the first place.

Both measures are now computed in the simulation and reported side by side for every model
(`reports/decision_<date>.md`):

- **Fill rate** = units shipped ÷ units demanded, pooled over units.
- **Cycle service level** = share of replenishment cycles with no stockout, pooled over cycles. A
  cycle opens when an order is placed and closes when it lands; it counts as a stockout cycle if
  any demand goes unmet in between — the exposure window `z · σ_LT` is sized to protect. A cycle
  still open when the horizon ends is not counted (an unfinished cycle is not evidence either
  way), and lost sales outside any open cycle — a series whose reorder point is so low it never
  orders — depress fill rate while belonging to no cycle at all. That asymmetry is real, not a
  bookkeeping artifact, and is part of why the two measures can diverge sharply here.

### Fill-rate-targeted policy (v1.1, second option)

`reorder_decision_fill_rate` sizes safety stock to hit a target *fill rate* directly, by inverting
the loss-function relationship above for `k` (`fill_rate_k`, Brent's method on `G`, with `Q`
estimated per series as its expected lead-time demand), rather than reading `z` off the normal
quantile. `reorder_point = lead_time_mean + k · σ_LT` exactly as before — only the safety factor
changes.

**Caveat on the fill-rate formula.** `β = 1 − σ_LT·G(k)/Q` is derived for a *fixed lot size* Q.
Under `S = s` there is no lot size — Q is the variable undershoot below the reorder point — and it
is measured, not assumed: realised Q has a coefficient of variation of 1.59 (see "How well does Q
behave?" in `reports/decision_<date>.md`). A formula handed a quantity that varies by more than its
own mean is partly being misapplied, so the gap between the fill rate it promises and the one the
simulation delivers cannot be attributed wholly to distributional shape. The CSL policy's `z · σ_LT`
makes no lot-size assumption and is unaffected.

Which policy is more conservative is **not fixed**: it turns on σ_LT/Q. A series whose lead-time
spread is small next to its per-cycle demand needs *less* than `z = 1.645` to reach a 95% fill
rate; a spiky series whose σ_LT dwarfs Q needs considerably more. Both regimes exist in this
panel, which is exactly why the CSL and fill-rate numbers cannot be read off one another. Both
policies are kept and reported — the comparison is the point — with the CSL policy remaining the
default for `make decide` and the serving path.

### Calibration schemes and the cost-optimal service level (v1.1)

`service_level_target` and the safety-stock calibration are two separate levers, both measured in
`reports/safety_stock_<date>.md` and `reports/service_level_sweep_<date>.md`, with paired-bootstrap
CIs over series.

**Calibration schemes** (`reorderpoint/safety_stock.py`) — how a bucket of pooled residuals becomes
a buffer. *Form*: `normal` (`z · σ_pooled`) or `empirical` (the residual distribution's own
quantile). *Granularity*: `intermittency` (the shipped 2 buckets), `volume_tercile_within_
intermittency` (formerly `intermittency_volume`; renamed because it populates 4 of its 6 theoretical
buckets — every `regular` series is already top-tercile by volume — so it only splits the
intermittent class), and `volume_quintile` (5 balanced buckets on volume alone). Granularity moves
cost by ~9–11%; form by at most ~1%. The shipped default is still `normal × intermittency`.

**Cost-optimal service level.** The newsvendor critical ratio `Cu / (Cu + Co)` is the *realised*
CSL that minimises expected cost. `Cu` is `stockout_penalty_per_unit`; `Co` is
`holding_cost_rate × unit_cost × period`, where `period` is how long a surplus unit is carried per
cycle it protects — the measured mean replenishment cycle (~9 days here), not the 7-day lead time,
which is the shortest possible cycle and so an upper bound on the ratio. `Co` scales with price, so
the ratio is per-SKU (p10 68% to p90 97.5% on this panel); a uniform target cannot be optimal for
all of them. The ratio is compared to *realised* CSL, not the nominal target `z` is computed from —
the two differ by ~12 points here, and comparing across them conflates the two.

Note that `holding_cost_rate` accrues **per day** (see "Cost simulation" and "Holding cost rate"
below). Every optimum computed here is a function of it: the cost-minimising target moves from ~92%
at the legacy 2%/day to the top of the grid at any real carrying cost.

## Cost simulation

Track A has no `unit_cost`/COGS field. **Mean training-window `price` is the per-series price
proxy**, and since 2026-09-20 the economics turn it into both sides of the trade (`CostParams`): a
lost sale costs `gross_margin x price`, and stock is valued at cost, `(1 - gross_margin) x price`
— see "Stockout penalty" below. (Before that: a flat $5 per lost unit, and stock valued at price.)

Per backtest fold, the decision (`s`, `S=s`) is computed once from the model's forecast at the
fold's origin (matching how `backtest.py` fits one model per fold, not one per day). The
simulation then assumes the SKU starts the fold **stocked to the reorder point** (`on_hand_start =
s` — a steady-state assumption: the business was already following the policy going into the
fold, so the comparison isn't dominated by an arbitrary cold-start stock level) and steps forward
one day at a time over realised demand:

- Demand is met from on-hand first; unmet demand is a **lost sale** (not backordered), costed at
  the unit's lost margin (`CostParams.stockout_penalty`).
- End-of-day on-hand accrues `holding_cost_rate * unit_cost * on_hand` — so the rate is a
  **per-day** fraction of unit cost: an annual carrying cost divided by 365. See "Holding cost rate"
  below for the value, its source, and why the model cost ranking depends on it.
- Whenever `on_hand < s` and no order is currently in transit, a new order for `S - on_hand`
  arrives after `lead_time_days` (single outstanding order at a time — continuous review).

Output per series per fold: total holding cost, total stockout cost, their sum, fill rate (units
shipped / units demanded), replenishment cycles and stockout cycles (which give the realised cycle
service level — see "Two service measures" above), and a simplified "turns" (units shipped /
average on-hand — not a true COGS-based turns ratio, since there's no COGS figure independent of
the price-as-cost proxy).

The headline table (`reports/decision_<date>.md`) aggregates cost across folds per model, same
structure as `reports/backtest_<date>.md`. Phase 4 acceptance: the best model's mean total cost is
lower than SeasonalNaive's, reported in USD.

## Holding cost rate

**Assumption (Track A):** carrying a unit costs **25% of its unit cost per year**, i.e.
`HOLDING_COST_RATE = 0.25 / 365 ≈ 0.000685` per day (simple, not compounded — the simulation accrues
it once per day). Set in `config.py` (`DEFAULT_ANNUAL_HOLDING_RATE`) and `.env.example`.

**Source:** Marc Goetschalckx, *Logistics Systems Design*, Chapter 5, "Inventory Systems" (Georgia
Institute of Technology course notes, 2002;
<https://www2.isye.gatech.edu/~mgoetsch/cali/logistics_systems_design/inventory_systems/inventory_systems.pdf>):
"A holding cost rate of 25 % of the unit value per year is a widely quoted average for the US
industry." Quote checked against the document itself.

**What that does and does not establish.** It is a cross-industry rule of thumb from teaching
notes, not a retail-specific survey, and it does not itself cite a measurement. I looked for a
retail-specific published figure with a primary source and did not find one I could read and
verify: the "retail carrying cost is 20–30%" figures on vendor blogs cite nothing, and the one
encyclopedia sentence to that effect is marked `[citation needed]`. Those are not cited here. So
this is an *assumption with a named source*, which is what the original "illustrative" 0.02
lacked, and not a retail benchmark. The 15–30%/yr *range* used to shade the chart
(`PLAUSIBLE_ANNUAL_HOLDING_RATE`) is the project's working range, not something that source
states; only the 25% is sourced.

**Why it matters, and why the choice of 25% is not doing the work.** Cost is exactly linear in the
rate (the policy is sized from the service level and never reads it, so the inventory trajectory is
identical at every rate — checked by re-simulating at 25%/yr, `reports/holding_breakeven_*.md` §2).
LightGBM and SeasonalNaive therefore cross at one rate: **~157%/yr** at the default cost model and
95% service target (95% bootstrap CI 84–243%; 364%/yr under the legacy flat-$5 penalty). Below it
SeasonalNaive is cheaper, above it LightGBM. Every value from 15% to 30% gives the same ranking, so
the decision does not turn on which figure inside that range is picked. The original 0.02/day was
~730%/yr, past the crossover, which is the only reason LightGBM ever looked cheapest. The crossover
is *not* a property of the holding rate alone: it scales with the lost-sale cost and depends on the
service target (~36%/yr at the cost-optimal target) — the next two sections.

**A real business:** replace it with its own carrying cost (capital cost + storage + shrink +
obsolescence, as an annual fraction of unit cost), divided by 365. Then `make breakeven` re-solves
the crossover, and the README figure and dashboard slider follow. Track B (UCI Online Retail II) is
public data with no cost information, so it uses this same 25%/yr as an assumption
(`docs/track_b.md`).

**Stored runs are not restated.** Every Phase 4 and ablation artifact on disk was simulated at the
legacy 0.02/day. New runs stamp the rate they used (`holding_cost_rate` column,
`decision._simulate_policy`); older ones are read as 0.02 (`holding_breakeven.simulated_rate`), and
re-pricing to any rate is exact, so nothing is re-run. The historical reports keep their numbers and
are labelled as legacy-rate results.

## Stockout penalty (lost margin)

The flat `$5.00` per lost unit was unsourced, and against a mean item price of ~$5.90 it priced a
lost sale at nearly the whole price of the item. What a retailer loses when it cannot sell a unit is
its margin. The default is now **a lost sale costs `gross_margin x price` and stock is valued at
`(1 - gross_margin) x price`**, one parameter (`GROSS_MARGIN`) tying both sides of the trade to the
same unit.

**Source:** Walmart Inc., Form 10-K for the fiscal year ended 2026-01-31 (SEC accession
0000104169-26-000055), **Walmart U.S. segment**: net sales $482,975M, gross profit $132,615M (net
sales less cost of sales) → gross profit rate **27.5%** (27.2% in fiscal 2025, 26.8% in 2024). The
consolidated Statements of Income give 24.2% (net sales $706,413M, cost of sales $535,395M); it
blends in Sam's Club and International and is kept as a sweep point (`CONSOLIDATED_GROSS_MARGIN`).
Both computed from the filing itself. (The first version of this section used the consolidated
figure, calling it "the right retailer"; the segment figure was in the same document and matches
the M5 data — Walmart U.S. stores in CA, TX and WI — better, so it replaced it.)

**What that does and does not establish.** M5 is Walmart U.S. store data, so the U.S. segment is
the matching figure; but it is a whole-segment average, not the hobby category, and Walmart's own
filing notes its cost of sales omits some distribution costs, so it is not strictly comparable to
other retailers. It is a **lower bound** on what a stockout costs: lost goodwill and demand that
does not return are extra. So it is an assumption with a named source, and the ranking's dependence
on it is reported as a sweep (`reports/penalty_sensitivity_*.md`), not trusted.

**Consequences.** (1) Under lost margin the critical ratio is the same for every SKU (both `Cu` and
`Co` scale with price), so a uniform service target is right under this cost model. (2) The
LightGBM/SeasonalNaive crossover is `h* ∝ margin / (1 − margin)` (verified on the data to 1e-15): 157%/yr
at the 27.5% default, 132%/yr at the consolidated 24.2%, 22%/yr at 5%, 2,350%/yr at 85% (≈ the old
flat $5). (3) At the default economics and 95% target SeasonalNaive is cheaper by $178 [$77, $299]
per fold at 25%/yr.

**Legacy reproduction.** Setting `STOCKOUT_PENALTY_PER_UNIT` in the environment selects the old flat
model. Stored runs carry the pricing they used; `economics.reprice` re-prices them per series to any
(holding rate, margin) exactly, checked against a real re-simulation.

## Service target

95% was a default, not a finding. The newsvendor critical ratio `CR = Cu / (Cu + Co)` is the cycle
service level that minimises expected cost: **98.8%** under the default economics with the 7-day
lead time as the protection period (98.4% with the measured 9-day cycle); 99.4% under the legacy
flat $5 at mean price, with the median SKU at 99.6%. `reports/optimal_target_*.md` extends the
nominal-target grid to 99.9999%: pooled cost is still 12% above its minimum at the old 99.9% edge,
is within 2% of it from 99.999%, and is minimal at the top of the grid (99.9999%, z = 4.75) — the
optimum is "at least 99.999%", not a sharp point (at the consolidated 24.2% margin it was interior at
99.999%). Cost falls from $1,569 to $949 per fold (−40%) between 95% and there.

**The optimum is not a target to ship.** Realised cycle service tops out at 96.8% even at 99.9999%
nominal — never reaching the critical ratio. Classifying every stockout cycle (replicating the
simulator's cycle logic exactly): at 95% nominal, 78% of stockout cycles are cases where lead-time
demand was *below* the reorder point and the stockout happened because on-hand had already dipped
under it when the order triggered; at 99.9999% it is 92%. Only 0.24% of cycles are genuine "demand
exceeded s" sizing failures. An `(s, S = s)` order triggers when on-hand falls *below* s, so the
policy never holds a full s of protection; a wider buffer only pays for what the trigger loses.
That is direct evidence that the order-up-to structure, not the safety-stock formula, limits
service (a one-off check of finer pooling and the empirical form, not reproduced by a module,
saturated at 94–97.6% too). The extreme nominal
optimum is a symptom of compensating for it. The earlier reading that the sweep's *grid edge* pointed
at `S = s` was wrong — the edge was the cost structure — but the saturation and this decomposition
are the evidence.

**The ranking depends on the target.** At 95% SeasonalNaive is cheaper than LightGBM by $178 [$77,
$299]; at the cost-optimal target the gap is +$44 [−$61, +$205] — indistinguishable — and the
crossover holding rate falls to 36%/yr. LightGBM is last of five in point estimate at every target
tested, but at the optimum none of its differences from the other four is distinguishable from
zero.

## Safety stock in serving

Serving sizes safety stock from the persisted calibration, not from the model's own interval width:
see `docs/serving.md` and `reorderpoint/calibration.py`. `decision.calibrated_reorder_decision` is
the single sizing rule shared with the backtest harness.

## Order-up-to level

`S = s` was a defective policy in this simulator, not just a simple one. An order replaces only the
deficit at the moment it is placed, and only one order is outstanding at a time, so demand that
arrives during the lead time is not covered until the *next* order. The stockout-cycle decomposition
(`Service target`, above) showed 78% of stockout cycles at 95% were cycles the reorder point would
have covered. `reorderpoint/order_up_to.py` tests the fix directly: `S = s + Q`, `Q` a multiple of
each series' expected lead-time demand (0, 0.5, 1, 2, 4), shipped and volume-quintile schemes,
seven targets from 90% to 99.9999%, all five models — `reports/order_up_to_*.md`.

**Result (shipped scheme, default economics, holding 25%/yr).** At the default 95% target, cost per
fold falls from **$1,569 to $453 at `S = s + 2 x lead-time demand`** (−$1,116 [−$1,920, −$576]),
realised CSL rises from 82.8% to 91.9%, fill rate from 81.5% to 97.5%, orders per fold fall from
1,239 to 674. The undershoot share of stockout cycles falls from **78% to 51%** at 95% and from
92% to 51% at each policy's own cost-optimal target: it moves substantially, but about half of
the remaining stockout cycles are still ones the reorder point would have covered. The cost-optimal
*nominal* target falls from ≥99.999% to **95%** (interior; the curve is flat from 90% to 99%), so
the default target is, for this policy, about right — and realised CSL at the extreme targets now
reaches ~99%, past the 98.8% critical ratio, instead of capping at 97%.

**Not an artefact of the starting stock.** A 28-day window holds only a cycle or two, so a policy
that starts higher in its cycle starts with more stock. The sweep starts each series at `s + Q/2`
(the steady-state average). Started at `s` — no extra initial stock — `S = s + 2x` costs $491; at
`S`, $485; at `s + Q/2`, $453; against $1,569 for `S = s` (`start_sensitivity`).

**The choice of `Q`, and its limits.** The multiple was chosen by a rule fixed before the results
were read: the lowest cost at the policy's own optimum, provided its CI against `S = s` excludes
zero. That is 2. It is *not* an EOQ: the simulation prices holding and lost sales but no fixed
cost per order, so a larger `Q` is charged for the stock it carries and credited only for the
service it buys. Median `Q` at multiple 2 is 4.6 units — the ordering cost under which that is the
EOQ is $0.06 per order — so any real ordering cost implies a *larger* economic lot; this lot is
conservative in that direction. The multiples 1 and 2 are statistically similar.

**What changes downstream.**
- **The model ranking reverses.** At `S = s` and 95% SeasonalNaive is cheaper than LightGBM by
  $178 [$77, $299]. At `S = s + 2x` LightGBM is cheaper by **$99 [$34, $160]**, with no holding-rate
  crossover at all (LightGBM has fewer stockouts *and* holds less), and 100% of bootstrap resamples
  at 15/25/30%/yr. So the earlier "SeasonalNaive is cheaper at any realistic carrying cost" was a
  property of the `S = s` policy — plausibly SeasonalNaive's higher forecasts acting as an unpriced
  safety margin against a policy that under-covered. Model choice depends on the ordering policy
  as well as on the holding rate, the lost-sale cost and the service target.
- **The quintile scheme's win does not survive.** Under `S = s` it passes the pre-registered
  held-out test (−$342 [−$711, −$109]); under `S = s + 2x` it fails criterion 1 (−$22 [−$58,
  +$8], crosses zero) though it passes 2 and 3. The default scheme therefore stays
  `normal x intermittency`. The calibration finding was tuned against a cost structure the lot
  size changes, as suspected.
- **Serving.** `decision.compute_decisions` takes `lot_multiple` and orders up to
  `S = s + Q` while on-hand is below s; `/reorder` reports `order_up_to`. The harness and serving
  are pinned to the same reorder point *and* order-up-to level (`tests/test_serving_calibration.py`).

**Caveats.** One simulator, one panel, and a baseline (`S = s` with a single outstanding order) that
is weak in a specific way; a stronger baseline — inventory-position triggering, or several orders
in flight — was not tested, and would shrink the gap. No ordering cost. `Q` scales with the
*forecast* lead-time demand, so it inherits the forecast's errors.
