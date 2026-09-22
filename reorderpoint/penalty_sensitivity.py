"""How much of the model cost ranking depends on the stockout penalty, and on the service target?

The holding rate was interrogated first (`holding_breakeven.py`) and turned out to decide the
LightGBM-vs-SeasonalNaive ranking. The stockout penalty is the other side of the same trade and
had the same defect: a flat `$5.00` per lost unit, unsourced, against a mean item price of ~$5.90
— i.e. a lost sale priced at nearly the whole price of the item, when what a retailer loses on a
lost sale is its margin. It now follows unit economics (`config.DEFAULT_GROSS_MARGIN`): a lost sale
costs `margin x price`, stock is valued at `(1 - margin) x price`.

Under those economics the LightGBM/SeasonalNaive crossover has a closed form. With `A` the holding
cost per unit rate per unit of cost basis and `B` the price-weighted lost units, cost is
`h(1-m)A + mB`, so the crossover holding rate is

    h*(m) = m / (1 - m) * (B_lgb - B_sn) / (A_sn - A_lgb)

— proportional to the penalty-to-cost ratio `m / (1-m)`. `scaling_check` verifies that on the data
rather than assuming it. The crossover also depends on the service target (tail behaviour differs
by model), so it is reported at the default 95% *and* at the cost-optimal target
(`optimal_target.py`).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from reorderpoint import backtest as bt
from reorderpoint import economics as econ
from reorderpoint import holding_breakeven as hb
from reorderpoint import optimal_target as ot
from reorderpoint.config import (
    CONSOLIDATED_GROSS_MARGIN,
    DEFAULT_ANNUAL_HOLDING_RATE,
    DEFAULT_GROSS_MARGIN,
    PLAUSIBLE_ANNUAL_HOLDING_RATE,
    annual_to_daily,
    daily_to_annual,
    load_config,
)

MARGINS = [0.05, 0.10, CONSOLIDATED_GROSS_MARGIN, DEFAULT_GROSS_MARGIN, 0.40, 0.60, 0.85, 0.95]
A, B = hb.MODEL_A, hb.MODEL_B


def _ci(r: pd.Series) -> str:
    span = f"[{r['ci_lo']:+,.0f}, {r['ci_hi']:+,.0f}]"
    return f"{r['point']:+,.0f} {span}" + (" — crosses zero" if r["crosses_zero"] else "")


def crossover_pct(detail: pd.DataFrame, margin: float | None) -> float | None:
    """LightGBM/SeasonalNaive crossover as annual % of unit cost, or None if they never cross."""
    be = hb.breakeven(hb.model_components(detail, gross_margin=margin), A, B)
    return None if be is None else daily_to_annual(be) * 100


def gap_at(detail: pd.DataFrame, margin: float | None, annual: float) -> float:
    """LightGBM minus SeasonalNaive, mean cost per fold, at `annual` holding."""
    c = hb.cost_at(hb.model_components(detail, gross_margin=margin), annual_to_daily(annual))
    return float(c[A] - c[B])


def scaling_check(detail: pd.DataFrame, margins: list[float] = MARGINS) -> float:
    """Max relative error of `h*(m) = kappa * m/(1-m)` against the directly solved crossover, with
    kappa taken from the default margin. ~0 confirms the closed form in the module docstring."""
    m0 = DEFAULT_GROSS_MARGIN
    kappa = crossover_pct(detail, m0) / (m0 / (1 - m0))
    errs = []
    for m in margins:
        direct = crossover_pct(detail, m)
        errs.append(abs(kappa * m / (1 - m) - direct) / direct)
    return float(max(errs))


def sensitivity_table(
    at95: pd.DataFrame, at_opt: pd.DataFrame, mean_price: float, margins: list[float] = MARGINS
) -> pd.DataFrame:
    rows = []
    for m in margins:
        row = {
            "gross margin": f"{m:.1%}"
            + (
                " (default, Walmart U.S.)"
                if m == DEFAULT_GROSS_MARGIN
                else " (Walmart consolidated)" if m == CONSOLIDATED_GROSS_MARGIN else ""
            ),
            "lost sale at mean price": f"${m * mean_price:.2f}",
        }
        for label, d in (("95%", at95), ("cost-optimal", at_opt)):
            x = crossover_pct(d, m)
            g = gap_at(d, m, DEFAULT_ANNUAL_HOLDING_RATE)
            row[f"crossover @ {label}"] = "none" if x is None else f"{x:,.0f}%/yr"
            row[f"LGB − SN @25%/yr, {label}"] = f"{g:+,.0f}"
            row[f"cheapest @25%/yr, {label}"] = (
                hb.cost_at(hb.model_components(d, gross_margin=m), annual_to_daily(0.25))
            ).idxmin()
        rows.append(row)
    return pd.DataFrame(rows)


# ---- chart -----------------------------------------------------------------------------------

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#e87ba4"


def plot_crossover_surface(
    at95: pd.DataFrame, at_opt: pd.DataFrame, legacy_pct: float, path
) -> None:
    """Crossover holding rate against the lost-sale margin, at the default and the cost-optimal
    target. Below a curve SeasonalNaive is cheaper; above it LightGBM is."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ms = np.linspace(0.02, 0.97, 200)
    lo, hi = (v * 100 for v in PLAUSIBLE_ANNUAL_HOLDING_RATE)
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(lo, hi, color=BAND, alpha=0.3, lw=0)
    for d, col, label, lw in (
        (at95, BLUE, "95% target (the default)", 2),
        (at_opt, ORANGE, "cost-optimal target", 2),
    ):
        k = crossover_pct(d, DEFAULT_GROSS_MARGIN) / (
            DEFAULT_GROSS_MARGIN / (1 - DEFAULT_GROSS_MARGIN)
        )
        ax.plot(ms, k * ms / (1 - ms), color=col, lw=lw, label=label, zorder=3)
        ax.plot(
            [DEFAULT_GROSS_MARGIN],
            [crossover_pct(d, DEFAULT_GROSS_MARGIN)],
            "o",
            ms=9,
            color=col,
            mec=SURFACE,
            mew=2,
            zorder=4,
        )
    ax.axhline(legacy_pct, color=INK_2, lw=1, ls=(0, (1, 3)))
    ax.text(
        0.03,
        legacy_pct * 1.08,
        f"legacy flat $5 penalty, 95% target: {legacy_pct:,.0f}%/yr",
        fontsize=9,
        color=INK_2,
    )
    ax.axvline(DEFAULT_GROSS_MARGIN, color=INK_2, lw=1, ls=(0, (1, 3)))
    ax.text(
        DEFAULT_GROSS_MARGIN + 0.01,
        1.35,
        f"Walmart U.S. FY2026\ngross margin {DEFAULT_GROSS_MARGIN:.1%}",
        fontsize=9,
        color=INK_2,
        va="bottom",
    )
    ax.text(
        0.5,
        20.5,
        "plausible carrying cost 15–30%/yr",
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
    )
    ax.set_yscale("log")
    ax.set_ylim(1, 3000)
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Cost of a lost sale, as gross margin (share of price)", color=INK_2)
    ax.set_ylabel("Holding rate where LightGBM = SeasonalNaive (per year)", color=INK_2)
    ax.text(0.3, 1700, "LightGBM cheaper above a curve", fontsize=9.5, color=INK)
    ax.text(0.62, 1.6, "SeasonalNaive cheaper below it", fontsize=9.5, color=INK)
    ax.legend(
        loc="upper left", frameon=False, labelcolor=INK, fontsize=10, bbox_to_anchor=(0, 0.97)
    )
    ax.grid(axis="y", color=GRID, lw=1)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.text(
        0.012,
        0.965,
        "The ranking depends on the penalty and the service target, too",
        color=INK,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.012,
        0.925,
        "Crossover ∝ margin/(1 − margin): verified against the data, not assumed.",
        color=INK_2,
        fontsize=10,
        va="top",
    )
    fig.subplots_adjust(top=0.88, left=0.1, right=0.97, bottom=0.11)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- report ----------------------------------------------------------------------------------


def build_report(
    at95: pd.DataFrame,
    at_opt: pd.DataFrame,
    opt_target: float,
    mean_price: float,
    chart_rel: str,
    sn95: pd.Series,
    snopt: pd.Series,
) -> tuple[str, dict]:
    m0 = DEFAULT_GROSS_MARGIN
    tbl = sensitivity_table(at95, at_opt, mean_price)
    legacy95 = crossover_pct(at95, None)
    x95, xopt = crossover_pct(at95, m0), crossover_pct(at_opt, m0)
    err = scaling_check(at95)
    top = PLAUSIBLE_ANNUAL_HOLDING_RATE[1] * 100
    sig95 = (not sn95["crosses_zero"]) and sn95["point"] > 0
    sigopt = (not snopt["crosses_zero"]) and snopt["point"] > 0
    if sig95:
        verdict = "**SeasonalNaive is significantly cheaper at the default target"
        verdict += (
            ", and at the cost-optimal one too.**"
            if sigopt
            else ", but the two are statistically indistinguishable at the cost-optimal target.**"
        )
    else:
        verdict = "**The two are statistically indistinguishable at the default target.**"
    verdict += (
        " At the default economics LightGBM is not significantly the cheapest at any plausible "
        "holding cost or target shown."
    )

    def boot_line(d: pd.DataFrame, label: str) -> str:
        priced = econ.reprice(d, DEFAULT_ANNUAL_HOLDING_RATE, m0)
        bb = hb.bootstrap_breakeven(priced, A, B, annual_rates=(0.15, 0.25, 0.30))
        ci = (
            f"[{bb.ci_lo * 100:,.0f}%, {bb.ci_hi * 100:,.0f}%]"
            if bb.ci_lo is not None
            else "not estimable"
        )
        shares = ", ".join(f"{r:.0%}: {p:.1%}" for r, p in bb.a_cheaper_at.items())
        return (
            f"- **{label}:** crossover {bb.point * 100:,.0f}%/yr, 95% CI {ci} "
            f"({bb.frac_undefined:.1%} of resamples had none); share of resamples where LightGBM "
            f"is cheaper at {shares}."
        )

    lines = [
        f"# Penalty and service-target sensitivity of the model ranking — {dt.date.today()}",
        "",
        "Two Track A parameters were unsourced and jointly decide the LightGBM/SeasonalNaive "
        "ranking: the holding rate (`reports/holding_breakeven_*.md`) and the stockout penalty, "
        "a flat $5.00 per lost unit against a mean item price of "
        f"${mean_price:.2f}. This report gives the penalty the same treatment, and adds the "
        "third quantity the ranking turns out to depend on, the service target.",
        "",
        "## 1. The penalty, tied to unit economics",
        "",
        "What a retailer loses when it cannot sell a unit is the margin on it, not the price. The "
        f"default is now a lost sale = **{m0:.1%} of price** (Walmart Inc., Form 10-K, fiscal year "
        "ended 2026-01-31, Walmart U.S. segment: net sales $482,975M, gross profit $132,615M; "
        "27.2% and 26.8% the two years before). The M5 data are Walmart U.S. stores, so the "
        "segment is the matching figure; the consolidated rate (24.2%: net sales $706,413M, cost "
        "of sales $535,395M) blends in Sam's Club and International and is a sweep point below. "
        "Neither is "
        "the hobby category, and Walmart's cost of sales omits some distribution costs. Stock is "
        "valued at cost, `(1 - margin) x price`, for the holding "
        f"rate. At the mean price that is ${m0 * mean_price:.2f} per lost unit, against the flat "
        f"$5.00. It is a **lower bound** on what a stockout costs — lost goodwill and demand that "
        "does not return are extra — which is why the margin is swept below rather than trusted.",
        "",
        f"![Crossover holding rate against the lost-sale margin]({chart_rel})",
        "",
        "## 2. The crossover surface",
        "",
        f"Holding {DEFAULT_ANNUAL_HOLDING_RATE:.0%}/yr for the gap columns; 'cost-optimal' is "
        f"{opt_target:.4%} nominal (`reports/optimal_target_*.md`), held fixed across margins. "
        "Positive gap = SeasonalNaive cheaper.",
        "",
        bt._markdown_table(tbl, float_cols=()),
        "",
        f"The closed form `h*(m) ∝ m/(1−m)` holds on the data to a maximum relative error of "
        f"{err:.1e} across margins 5%–95%.",
        "",
        "## 3. What survives, and what does not",
        "",
        f"- **At the default 95% target and default economics the crossover is {x95:,.0f}%/yr** "
        f"({x95 / top:.1f}× the top of the plausible range), against {legacy95:,.0f}%/yr under "
        "the legacy flat $5 penalty. SeasonalNaive is cheaper across the plausible holding range "
        "in both — but the '12× outside the range' framing was specific to the flat penalty.",
        f"- **At the cost-optimal target it is {xopt:,.0f}%/yr** — barely above the plausible "
        f"range's {top:.0f}% ceiling. Where the gap at 25%/yr and its interval sit there is in "
        "`reports/optimal_target_*.md` §3; the point is that the ranking is no longer robust.",
        "- **At margins of ~10% and below the ranking flips inside the plausible holding range** "
        "at "
        "the cost-optimal target. A lost sale costing under a tenth of the item's price would sit "
        f"well below Walmart U.S.'s own {m0:.1%} gross margin, so that is the edge of the region, "
        "not a "
        "likely case.",
        boot_line(at95, "95% target, default economics"),
        boot_line(at_opt, f"{opt_target:.4%} target, default economics"),
        "",
        'So the defensible statement is not "the naive baseline wins at any real cost" but a '
        "conditional one, stated with the intervals at the default economics and 25%/yr "
        "(LightGBM minus SeasonalNaive per fold, paired bootstrap):",
        "",
        f"- default 95% target: {_ci(sn95)}",
        f"- cost-optimal {opt_target:.4%} target: {_ci(snopt)}",
        "",
        verdict,
        "",
    ]
    return "\n".join(lines), {"x95": x95, "xopt": xopt, "legacy95": legacy95, "scaling_err": err}


def main() -> None:
    from reorderpoint import decision as dec

    costs = load_config().costs
    detail = pd.read_parquet(ot.DETAIL_PATH)
    c = ot.curves(ot.priced(detail, DEFAULT_ANNUAL_HOLDING_RATE, costs.gross_margin))
    opt = ot.argmin_target(c)["target"]
    at95 = detail[ot.at_target(detail, ot.REF_TARGET)]
    at_opt = detail[ot.at_target(detail, opt)]

    # the sweep's 95% rows and the stored decision backtest are the same simulation
    stored = pd.read_parquet(dec.DETAIL_PATH)
    if abs(crossover_pct(stored, None) - crossover_pct(at95, None)) > 1e-6:
        raise SystemExit("target sweep at 95% disagrees with the stored decision backtest")

    panel = bt.load_eval_panel()
    last_train = panel[panel["date"] <= bt.make_folds(panel)[-1].train_end]
    mean_price = float(dec.unit_cost_from_price(last_train).mean())

    files = bt.REPORTS_DIR / "penalty_sensitivity_files"
    files.mkdir(parents=True, exist_ok=True)
    plot_crossover_surface(
        at95, at_opt, crossover_pct(at95, None), files / "crossover_vs_margin.png"
    )
    p_all = ot.priced(detail, DEFAULT_ANNUAL_HOLDING_RATE, costs.gross_margin)
    sn = lambda t: ot.compare_at(p_all, t).query("other == 'SeasonalNaive'").iloc[0]  # noqa: E731
    report, stats = build_report(
        at95,
        at_opt,
        opt,
        mean_price,
        "penalty_sensitivity_files/crossover_vs_margin.png",
        sn(ot.REF_TARGET),
        sn(opt),
    )
    out = bt.REPORTS_DIR / f"penalty_sensitivity_{dt.date.today().isoformat()}.md"
    out.write_text(report)
    print(f"Wrote {out}")
    print(stats)


if __name__ == "__main__":
    main()
