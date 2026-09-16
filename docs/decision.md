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

**3. Order quantity.** This project uses an order-up-to policy with `S = s` — no separate EOQ
lot-sizing (a deliberate simplification; that refinement isn't needed to answer the questions
this project asks). When `on_hand < s`:

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

`Q` is the demand consumed per replenishment cycle. Because this project sets `S = s` (step 3
below), `Q` is only the deficit accumulated since the last order — *small*. Fill rate depends on
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

## Cost simulation

Track A has no `unit_cost`/COGS field, so **mean training-window `price` stands in for unit
cost** — the same kind of proxy caveat as `y`-as-demand (`docs/data.md`).

Per backtest fold, the decision (`s`, `S=s`) is computed once from the model's forecast at the
fold's origin (matching how `backtest.py` fits one model per fold, not one per day). The
simulation then assumes the SKU starts the fold **stocked to the reorder point** (`on_hand_start =
s` — a steady-state assumption: the business was already following the policy going into the
fold, so the comparison isn't dominated by an arbitrary cold-start stock level) and steps forward
one day at a time over realised demand:

- Demand is met from on-hand first; unmet demand is a **lost sale** (not backordered), costed at
  `stockout_penalty_per_unit`.
- End-of-day on-hand accrues `holding_cost_rate * unit_cost * on_hand`.
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
