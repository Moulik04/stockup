"""v1.1 Task 7: is `service_level_target = 0.95` cost-optimal for the configured costs?

The simulation already prices both sides of the trade — `stockout_penalty_per_unit` per lost unit,
`holding_cost_rate * unit_cost` per unit-day on hand — so two questions can be answered from the
project's own numbers instead of an arbitrary round target:

1. **What does newsvendor theory say?** The critical ratio `Cu / (Cu + Co)` is the *realised*
   cycle service level that minimises expected cost. `Cu` is the stockout penalty per unit; `Co`
   is what one surplus unit costs to carry. Here holding accrues per *day*, so `Co` needs a period:
   the replenishment cycle, since a buffer unit sits on the shelf for the whole cycle it protects.
   The cycle length is measured from the simulation (series-days per order) rather than assumed;
   the lead time is reported alongside as the shortest possible cycle, i.e. the upper bound on CR.
   `Co` scales with price, so CR is really a per-SKU quantity — the distribution is reported, not
   just the value at the mean price.
2. **What does the simulation say?** The shipped scheme (and the ablation's best scheme) swept over
   nominal targets 0.80–0.99, same forecasts, folds, models and calibration as everything else.

The two are compared on the right axis. The critical ratio is a statement about *realised* CSL;
the sweep's x-axis is the *nominal* target `z` is computed from, and realised CSL sits ~12 points
below nominal here. Comparing CR to the nominal argmin alone would conflate the two.

Two branches were fixed before running, per the v1.1 scope note:
- implied optimum well below 95% -> the cost function never agreed with the 95% target;
- implied optimum near 95% while cost falls as service degrades -> a pricing bug; find it first.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import decision as dec
from reorderpoint import divergence as dv
from reorderpoint.config import CostParams, daily_to_annual, describe_holding_rate, load_config

# 0.995 / 0.999 extend the grid past 0.99: the optimum moves up as holding gets cheaper
# (holding_sensitivity.py), and a grid that stops at 0.99 would clip it and report the edge.
LEVELS = [round(x, 2) for x in np.arange(0.80, 0.9901, 0.01)] + [0.995, 0.999]
SWEEP_CELLS = [ab.BASELINE_CELL, ("normal", "volume_quintile")]
SWEEP_PATH = bt.PANEL_PATH.parent / "service_level_sweep.parquet"
# Per (cell, target, fold, model): every cost component, kept so later analyses can re-price the
# same simulation (holding is linear in its rate and never enters the policy) or hold a fold out.
SWEEP_ROWS_PATH = bt.PANEL_PATH.parent / "service_level_sweep_rows.parquet"
# below this many points under the configured target the optimum counts as "well below"
WELL_BELOW = 0.05


def critical_ratio(costs: CostParams, price: float | np.ndarray, period_days: float):
    """Cu / (Cu + Co), with Co = holding_cost_rate * unit_cost * period_days.

    `price` is the mean training-window price; `costs` turns it into the per-unit lost-sale cost
    (`Cu`) and the value stock is held at (`unit_cost`) — under unit economics both scale with
    price, under the legacy flat penalty only `Co` does.

    `period_days` is how long a surplus unit is carried to protect one replenishment cycle.
    """
    price = np.asarray(price, dtype=float)
    co = costs.holding_cost_rate * costs.unit_cost(price) * period_days
    cu = costs.stockout_penalty(price)
    return cu / (cu + co)


def realised_cycle_days(detail: pd.DataFrame, horizon_days: int) -> float:
    """Mean days between orders: series-days simulated / orders placed."""
    return float(horizon_days * len(detail) / detail["orders"].sum())


def run_sweep(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    calib: pd.DataFrame,
    costs: CostParams,
    cells: list[tuple[str, str]] = SWEEP_CELLS,
    levels: list[float] = LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (curve, cycle_detail, fold_rows): mean cost/service per (cell, target), per-target
    realised cycle length, and every (cell, target, fold, model) row. Means are across folds and
    models, as everywhere else in the ablation.
    """
    rows, cycle_rows, fold_rows = [], [], []
    for form, granularity in cells:
        for level in levels:
            # n_boot only sizes the bucket-statistic CIs, which the normal form's point estimate
            # never depends on — keep it small, the sweep runs it dozens of times.
            results, detail, _ = ab.run_cell(
                panel, forecasts, calib, form, granularity, costs, n_boot=20, service_level=level
            )
            fold_rows.append(
                results.assign(
                    form=form,
                    granularity=granularity,
                    target=level,
                    holding_cost_rate=costs.holding_cost_rate,
                )
            )
            rows.append(
                {
                    "form": form,
                    "granularity": granularity,
                    "target": level,
                    **results[
                        [
                            "total_cost",
                            "holding_cost",
                            "stockout_cost",
                            "fill_rate",
                            "cycle_service_level",
                        ]
                    ].mean(),
                }
            )
            cycle_rows.append(
                {
                    "form": form,
                    "granularity": granularity,
                    "target": level,
                    "cycle_days": realised_cycle_days(detail, bt.HORIZON),
                }
            )
            print(f"sweep: {form} x {granularity} @ {level:.2f}")
    return (
        pd.DataFrame(rows),
        pd.DataFrame(cycle_rows),
        pd.concat(fold_rows, ignore_index=True),
    )


def dominance_check(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    calib: pd.DataFrame,
    costs: CostParams,
    curve: pd.DataFrame,
) -> dict | None:
    """Does the finer scheme beat the shipped one at *matched realised service*, not just at the
    same nominal target?

    Cost falling while CSL falls looks impossible in a simulation that prices stockouts, and is
    impossible along one scheme's target axis (more buffer -> higher CSL, always). Across schemes
    it is not: CSL counts cycles, cost counts lost units and their price, so a scheme that moves
    buffer toward high-volume SKUs can cut cost and CSL together. The test that separates that
    from a pricing bug is to raise the finer scheme's target until its realised CSL is at least
    the baseline's, and see whether it is *still* cheaper. The target is chosen by that service
    criterion (smallest grid target that matches baseline CSL), not by minimising cost, so the
    comparison is not tuned toward the answer.
    """
    target = costs.service_level_target
    base_cell, fine_cell = ab.BASELINE_CELL, ("normal", "volume_quintile")
    base_curve = curve[(curve.form == base_cell[0]) & (curve.granularity == base_cell[1])]
    fine_curve = curve[(curve.form == fine_cell[0]) & (curve.granularity == fine_cell[1])]
    base_row = base_curve[np.isclose(base_curve["target"], target)].iloc[0]
    matched = fine_curve[fine_curve["cycle_service_level"] >= base_row["cycle_service_level"]]
    if matched.empty:
        return None
    matched_target = float(matched.sort_values("target").iloc[0]["target"])

    _, base_detail, _ = ab.run_cell(
        panel, forecasts, calib, *base_cell, costs, n_boot=20, service_level=target
    )
    _, fine_detail, _ = ab.run_cell(
        panel, forecasts, calib, *fine_cell, costs, n_boot=20, service_level=matched_target
    )
    tag = lambda d: d.assign(cycles_ok=d["cycles"] - d["stockout_cycles"])  # noqa: E731
    base_detail, fine_detail = tag(base_detail), tag(fine_detail)
    groups = ["fold", "model"]
    return {
        "matched_target": matched_target,
        "fine_row": fine_curve[np.isclose(fine_curve["target"], matched_target)].iloc[0],
        "base_row": base_row,
        "cost": boot.bootstrap_diff(
            fine_detail, base_detail, value_col="total_cost", group_cols=groups
        ),
        "csl": boot.bootstrap_ratio_diff(
            fine_detail, base_detail, "cycles_ok", "cycles", group_cols=groups
        ),
        "fill_rate": boot.bootstrap_ratio_diff(
            fine_detail, base_detail, "units_shipped", "units_demanded", group_cols=groups
        ),
    }


def plot_curve(curve: pd.DataFrame, cr_lines: dict[str, float], target: float, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for (form, gran), g in curve.groupby(["form", "granularity"]):
        g = g.sort_values("target")
        label = f"{form} × {gran}"
        axes[0].plot(g["target"], g["total_cost"], marker="o", ms=3, lw=1.6, label=label)
        axes[1].plot(
            g["cycle_service_level"], g["total_cost"], marker="o", ms=3, lw=1.6, label=label
        )
    axes[0].axvline(target, color="0.4", ls="--", lw=1, label=f"configured target {target:.0%}")
    axes[0].set_xlabel("nominal service-level target (z = Φ⁻¹(target))")
    axes[0].set_ylabel("mean total cost per fold ($)")
    axes[0].set_title("Cost vs nominal target")
    axes[1].set_xlabel("realised cycle service level")
    axes[1].set_title("Cost vs realised CSL")
    for name, value in cr_lines.items():
        axes[1].axvline(value, ls=":", lw=1.2, color="crimson" if "lead" in name else "darkorange")
        axes[1].text(value, axes[1].get_ylim()[1], f" {name}", rotation=90, va="top", fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build_report(
    curve: pd.DataFrame,
    cycle_detail: pd.DataFrame,
    costs: CostParams,
    unit_costs: pd.Series,
    plot_rel_path: str | None,
    dominance: dict | None = None,
) -> tuple[str, dict]:
    target = costs.service_level_target
    lead = costs.lead_time_days
    mean_uc = float(unit_costs.mean())

    base_cell = ab.BASELINE_CELL
    base = curve[(curve.form == base_cell[0]) & (curve.granularity == base_cell[1])].sort_values(
        "target"
    )
    cycle_len = float(
        cycle_detail[
            (cycle_detail.form == base_cell[0])
            & (cycle_detail.granularity == base_cell[1])
            & (np.isclose(cycle_detail.target, target))
        ]["cycle_days"].iloc[0]
    )

    cr_lead = float(critical_ratio(costs, mean_uc, lead))
    cr_cycle = float(critical_ratio(costs, mean_uc, cycle_len))
    per_series_cr = critical_ratio(costs, unit_costs.to_numpy(), cycle_len)
    q = np.quantile(per_series_cr, [0.1, 0.25, 0.5, 0.75, 0.9])

    def argmin(df: pd.DataFrame) -> pd.Series:
        return df.loc[df["total_cost"].idxmin()]

    best_base = argmin(base)
    at_target = base[np.isclose(base["target"], target)].iloc[0]
    saving = (at_target["total_cost"] - best_base["total_cost"]) / at_target["total_cost"]

    # is the curve U-shaped, or does cost keep falling as the target is relaxed? Compare the
    # cheapest point against the two ends of the sweep.
    lo_end, hi_end = base.iloc[0], base.iloc[-1]
    u_shaped = best_base["total_cost"] < min(lo_end["total_cost"], hi_end["total_cost"]) - 1e-9

    gap_nominal = target - best_base["target"]
    if not u_shaped:
        verdict_kind = "monotone"
    elif gap_nominal > WELL_BELOW:
        verdict_kind = "well_below"
    elif gap_nominal < 0.01:
        verdict_kind = "near"
    else:
        verdict_kind = "below"

    table = base[
        [
            "target",
            "total_cost",
            "holding_cost",
            "stockout_cost",
            "fill_rate",
            "cycle_service_level",
        ]
    ].rename(columns={"total_cost": "mean_total_cost"})
    table["vs_target"] = (table["mean_total_cost"] / at_target["total_cost"] - 1) * 100

    quint = ("normal", "volume_quintile")
    qcurve = curve[(curve.form == quint[0]) & (curve.granularity == quint[1])].sort_values("target")
    best_q = argmin(qcurve)
    q_at_target = qcurve[np.isclose(qcurve["target"], target)].iloc[0]

    # The 2%/day default (~730%/yr) made every conclusion here conditional on a mispriced holding
    # term; the caveats below are only true at a rate that high, so they are only printed then.
    implausible = daily_to_annual(costs.holding_cost_rate) > 1.0
    if implausible:
        rate_caveat = [
            "> **Read everything below as conditional on the holding rate.** It is priced per "
            f"*day* (~{daily_to_annual(costs.holding_cost_rate):.0%}/yr), its intended unit was "
            "never recorded, and the conclusions here do not survive plausible alternatives: at "
            "2%/month or 2%/year the cost-minimising target is at the top of the grid, not ~92%. "
            "See `reports/holding_cost_sensitivity_<date>.md`. The 'optimum is only 3 points "
            "below target' reading is a statement about 2%/day.",
            "",
        ]
        unit_check = [
            "**Unit check worth confirming:** holding accrues per day, so "
            f"{costs.holding_cost_rate:.0%} is ~{daily_to_annual(costs.holding_cost_rate):.0%} of "
            "unit cost per year. If `HOLDING_COST_RATE` was meant as an annual or monthly rate, "
            "holding is overpriced by 30–365× and every optimum here is biased *downward*.",
            "",
        ]
    else:
        rate_caveat = [
            "> Holding is priced at a plausible annual carrying cost "
            "(`reports/holding_breakeven_<date>.md`). The service-level sweeps written before "
            "2026-09-20 used 2%/day (~730%/yr) and are not comparable: their optimum (~92%) was "
            "a statement about that rate.",
            "",
        ]
        unit_check = []

    lines = [
        f"# Cost-optimal service level — {dt.date.today().isoformat()}",
        "",
        f"Configured costs: stockout penalty ${costs.stockout_penalty_per_unit:.2f}/unit, holding "
        f"{describe_holding_rate(costs.holding_cost_rate)} of unit cost, lead time {lead} days, "
        f"target {target:.0%}. Mean unit cost (training-window price) ${mean_uc:.2f}.",
        "",
        *rate_caveat,
        "## 1. What the critical ratio says",
        "",
        "`CR = Cu / (Cu + Co)` is the *realised* cycle service level that minimises expected cost, "
        "with `Cu` = stockout penalty per unit and `Co` = `holding_cost_rate × unit_cost × "
        "period`, `period` being how long a surplus unit is carried per cycle it protects.",
        "",
        "| period used for Co | days | CR at mean price |",
        "| --- | --- | --- |",
        f"| lead time (shortest possible cycle; upper bound on CR) | {lead} | {cr_lead:.1%} |",
        f"| measured mean cycle (series-days per order, target {target:.0%}) | {cycle_len:.1f} "
        f"| {cr_cycle:.1%} |",
        "",
        f"Because `Co` scales with price, CR is a per-SKU quantity. Across the {len(unit_costs)} "
        f"series (measured cycle length), CR runs p10 {q[0]:.1%} / p25 {q[1]:.1%} / **median "
        f"{q[2]:.1%}** / p75 {q[3]:.1%} / p90 {q[4]:.1%}: cheap SKUs justify near-universal "
        "service, expensive ones a much lower one. **A single uniform target cannot be optimal "
        "for all of them** — a separate finding from where the uniform optimum lands.",
        "",
        *unit_check,
        "## 2. What the simulation says",
        "",
        f"Shipped scheme (`{base_cell[0]} × {base_cell[1]}`), nominal target swept "
        f"{LEVELS[0]:.2f}–{LEVELS[-1]:.2f}, means across folds and models:",
        "",
        bt._markdown_table(
            table,
            float_cols=(
                "target",
                "mean_total_cost",
                "holding_cost",
                "stockout_cost",
                "fill_rate",
                "cycle_service_level",
                "vs_target",
            ),
        ),
        "",
        "`vs_target` = % change in cost relative to the configured target's row.",
        "",
    ]
    if plot_rel_path:
        lines += [f"![Cost vs service level]({plot_rel_path})", ""]

    realised_at_min = best_base["cycle_service_level"]
    lines += [
        f"Cost is minimised at a nominal target of **{best_base['target']:.0%}** "
        f"(${best_base['total_cost']:,.0f}/fold), {saving:+.1%} vs. the configured "
        f"{target:.0%} (${at_target['total_cost']:,.0f}). Realised CSL there is "
        f"{realised_at_min:.1%}, fill rate {best_base['fill_rate']:.1%}.",
        "",
        "**Compare on the right axis.** The critical ratio is a statement about realised CSL "
        f"({cr_cycle:.1%} at the mean price, {q[2]:.1%} median across SKUs); the shipped policy at "
        f"the configured {target:.0%} target already *realises* "
        f"{at_target['cycle_service_level']:.1%}. Against the realised axis the 'gap' the project "
        "has been describing is far smaller than 12 points — see the verdict.",
        "",
        "## 3. Verdict (branches fixed in advance)",
        "",
    ]

    if verdict_kind == "well_below":
        lines += [
            f"**Implied optimum well below the target.** Simulated cost bottoms out at a nominal "
            f"{best_base['target']:.0%}, {gap_nominal * 100:.0f} points under the configured "
            f"{target:.0%}, and the critical ratio independently puts the realised optimum near "
            f"{cr_cycle:.0%}. The cost function never agreed with a 95% target. That is not an "
            "error in the simulation — it is the simulation correctly pricing what the target "
            "ignores.",
        ]
    elif verdict_kind == "near":
        lines += [
            f"**Implied optimum near the target ({best_base['target']:.0%} vs {target:.0%}), and "
            "the curve is U-shaped:** cost rises on both sides, so there is no evidence that cost "
            "improves as service degrades, and no pricing bug is indicated by the shape. "
            f"Configured target and simulated optimum agree to within a point; the branch that "
            "would call for a bug hunt does not trigger.",
        ]
    elif verdict_kind == "below":
        lines += [
            f"**Optimum below the target, but not by a wide margin** "
            f"({best_base['target']:.0%} vs {target:.0%}, {gap_nominal * 100:.0f} points). Neither "
            "pre-registered branch fires cleanly; the curve is U-shaped, so nothing looks like a "
            "pricing bug, and the target is mildly too high for these costs.",
        ]
    else:
        lines += [
            "**Cost falls monotonically across the sweep** — no interior optimum. That is the "
            "signature the pricing-bug branch exists for; it must be investigated before anything "
            "ships.",
        ]

    flat = float(
        (
            base[
                (base.target >= best_base.target - 0.03) & (base.target <= best_base.target + 0.03)
            ]["total_cost"].max()
            - best_base["total_cost"]
        )
        / best_base["total_cost"]
    )
    lines += [
        "",
        f"The curve is flat near its minimum: within ±3 points of nominal target the cost stays "
        f"within {flat:.1%} of the minimum, so the argmin itself is not sharply determined and "
        "should be read as a region, not a number.",
        "",
        "## 4. Does the optimum depend on the calibration scheme?",
        "",
        f"Same sweep on the ablation's best cell (`{quint[0]} × {quint[1]}`): minimum at a nominal "
        f"**{best_q['target']:.0%}** (${best_q['total_cost']:,.0f}/fold, realised CSL "
        f"{best_q['cycle_service_level']:.1%}); at the configured {target:.0%} it costs "
        f"${q_at_target['total_cost']:,.0f}. ",
        "",
        bt._markdown_table(
            qcurve[["target", "total_cost", "fill_rate", "cycle_service_level"]].rename(
                columns={"total_cost": "mean_total_cost"}
            ),
            float_cols=("target", "mean_total_cost", "fill_rate", "cycle_service_level"),
        ),
        "",
    ]
    if dominance is not None:
        d, b, f = dominance, dominance["base_row"], dominance["fine_row"]

        def ci(row: dict, scale: float = 1.0, prefix: str = "", suffix: str = "") -> str:
            sign = "+" if row["point"] >= 0 else ""
            return (
                f"{sign}{prefix}{row['point'] * scale:,.1f}{suffix} "
                f"[{prefix}{row['ci_lo'] * scale:,.1f}{suffix}, "
                f"{prefix}{row['ci_hi'] * scale:,.1f}{suffix}]"
                + (" — crosses zero" if row["crosses_zero"] else "")
            )

        dominated = (
            not d["cost"]["crosses_zero"]
            and d["cost"]["point"] < 0
            and (d["csl"]["crosses_zero"] or d["csl"]["point"] > 0)
            and (d["fill_rate"]["crosses_zero"] or d["fill_rate"]["point"] > 0)
        )
        fill_note = (
            "not distinguishably different"
            if d["fill_rate"]["crosses_zero"]
            else "better (CI excludes zero)"
        )
        lines += [
            "## 5. Is cost-down-with-CSL-down a pricing bug? No: compare at matched service",
            "",
            "Cost falling while CSL falls looks impossible in a simulation that prices stockouts. "
            "Along a single scheme's target axis it *is* impossible, and the tables above show "
            "it: realised CSL rises monotonically with the target while cost is U-shaped. Across "
            "schemes it is not, because CSL counts every replenishment cycle equally while cost "
            "counts lost units and their price — a scheme that moves buffer toward high-volume "
            "SKUs cuts cost and CSL together. The test that separates that from a bug is to raise "
            "the finer scheme's target until its realised CSL is at least the baseline's and see "
            "whether it is still cheaper. The target is picked by that service criterion (the "
            "smallest grid target that matches baseline CSL), not by minimising cost.",
            "",
            f"`normal × volume_quintile` at nominal **{d['matched_target']:.0%}** (realised CSL "
            f"{f['cycle_service_level']:.1%}, fill rate {f['fill_rate']:.1%}) vs. the shipped "
            f"`normal × intermittency` at {target:.0%} (CSL {b['cycle_service_level']:.1%}, fill "
            f"rate {b['fill_rate']:.1%}). Paired bootstrap over series, 2,000 resamples, "
            "fine minus baseline:",
            "",
            f"- **Cost**: {ci(d['cost'], prefix='$')} per fold",
            f"- **Cycle service level**: {ci(d['csl'], scale=100, suffix='pp')}",
            f"- **Fill rate**: {ci(d['fill_rate'], scale=100, suffix='pp')}",
            "",
            (
                "**At matched service the finer scheme is still cheaper, with cycle service level "
                "not distinguishably different and fill rate " + fill_note + ".** The cost/CSL "
                "trade-off in the ablation was an artefact of comparing the two schemes at the "
                "same *nominal* target, where they realise different service. There is no pricing "
                "bug, and no trade-off to adjudicate: the finer scheme at a slightly higher "
                "nominal target is cheaper, matches on CSL and is no worse on fill rate."
                if dominated
                else "**The finer scheme does not cleanly dominate at matched service** — see the "
                "intervals above; the trade-off is not fully dissolved by matching CSL."
            ),
            "",
            "One caveat: the matched target is read off the same folds it is then evaluated on. "
            "It is one parameter chosen from a 20-point grid by a service criterion, so the "
            "optimism is small, but a fresh time window would be the clean test.",
            "",
        ]
    stats = {
        "cr_lead": cr_lead,
        "cr_cycle": cr_cycle,
        "cr_median": float(q[2]),
        "cycle_days": cycle_len,
        "argmin_target": float(best_base["target"]),
        "argmin_realised_csl": float(realised_at_min),
        "saving_vs_target": float(saving),
        "kind": verdict_kind,
        "q_argmin_target": float(best_q["target"]),
        "u_shaped": bool(u_shaped),
    }
    return "\n".join(lines), stats


def main() -> None:
    panel = bt.load_eval_panel()
    costs = load_config().costs

    forecasts = pd.read_parquet(dv.FORECASTS_PATH)
    calib = ab.calibration_residuals(panel, costs)

    curve, cycle_detail, fold_rows = run_sweep(panel, forecasts, calib, costs)
    curve.to_parquet(SWEEP_PATH, index=False)
    fold_rows.to_parquet(SWEEP_ROWS_PATH, index=False)

    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    last_train = eval_panel[eval_panel["date"] <= bt.make_folds(eval_panel)[-1].train_end]
    unit_costs = dec.unit_cost_from_price(last_train)

    date_str = dt.date.today().isoformat()
    files_dir = bt.REPORTS_DIR / "service_level_files"
    files_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"cost_vs_service_level_{date_str}.png"

    base = ab.BASELINE_CELL
    cycle_len = float(
        cycle_detail[
            (cycle_detail.form == base[0])
            & (cycle_detail.granularity == base[1])
            & np.isclose(cycle_detail.target, costs.service_level_target)
        ]["cycle_days"].iloc[0]
    )
    cr_lines = {
        f"CR (lead {costs.lead_time_days}d)": float(
            critical_ratio(costs, unit_costs.mean(), costs.lead_time_days)
        ),
        f"CR (cycle {cycle_len:.1f}d)": float(critical_ratio(costs, unit_costs.mean(), cycle_len)),
    }
    plot_curve(curve, cr_lines, costs.service_level_target, files_dir / plot_name)

    print("dominance check...")
    dominance = dominance_check(panel, forecasts, calib, costs, curve)
    report, stats = build_report(
        curve, cycle_detail, costs, unit_costs, f"service_level_files/{plot_name}", dominance
    )
    out_path = bt.REPORTS_DIR / f"service_level_sweep_{date_str}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print(stats)


if __name__ == "__main__":
    main()
