"""At what holding-cost rate does the model cost ranking flip, and where do real carrying costs sit?

The Phase 4 headline — LightGBM has the lowest simulated cost — was measured at
`holding_cost_rate` = 0.02 **per day** (~730% of unit cost per year). `holding_sensitivity.py`
showed the ranking against SeasonalNaive reverses at 2%/month and 2%/year. This module finishes the
job: it *solves* for the rate at which the two cost curves cross, and publishes the whole curve
instead of three points.

**Why the curve is exactly two straight lines.** The safety-stock policy is sized from the service
level and the forecast residuals; it never reads the holding rate (`docs/decision.md`). So the
inventory trajectory — orders, stock on hand, stockouts — is the same whatever the rate is, and a
model's cost is `holding_per_rate * rate + stockout`, linear in the rate with an intercept that does
not move. Two models' costs are two lines; they cross at most once, at

    rate* = (stockout_a - stockout_b) / (holding_per_rate_b - holding_per_rate_a)

That closed form is the answer. Nothing is fitted. `main()` also checks it three independent ways:
a numeric root-find on the re-priced raw rows, a full re-simulation of the shipped policy at another
rate through the real simulation code (trajectory identical, holding exactly linear), and a paired
bootstrap over series for the interval on the crossover.

Rates are per **day**, as the simulation accrues them; every user-facing figure is annualised
(x365, simple) because that is the unit carrying costs are quoted in.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reorderpoint import bootstrap as boot
from reorderpoint import economics as econ
from reorderpoint.config import (
    DEFAULT_ANNUAL_HOLDING_RATE,
    PLAUSIBLE_ANNUAL_HOLDING_RATE,
    annual_to_daily,
    daily_to_annual,
)
from reorderpoint.economics import LEGACY_SIMULATED_RATE, simulated_rate  # noqa: F401 (re-export)

MODEL_A = "LightGBM"
MODEL_B = "SeasonalNaive"


def _csl(detail: pd.DataFrame) -> pd.DataFrame:
    return detail[detail["policy"] == "csl"] if "policy" in detail.columns else detail


_AS_SIMULATED = object()


def model_components(
    detail: pd.DataFrame,
    gross_margin: float | None | object = _AS_SIMULATED,
    flat_penalty: float = econ.LEGACY_FLAT_PENALTY,
) -> pd.DataFrame:
    """Per model, mean over folds of the fold's summed cost, split into its two parts.

    Columns: `holding_per_rate` (holding cost per fold at a daily rate of 1.0, so cost at any rate
    is `holding_per_rate * rate + stockout`) and `stockout` (independent of the rate).

    By default the split is taken as the run was priced. Pass `gross_margin` to re-price every
    series first (`economics.reprice`): a float for unit economics (lost sale = margin x price,
    stock at cost), or None for the legacy flat `flat_penalty`."""
    d = _csl(detail)
    if gross_margin is _AS_SIMULATED:
        sim = simulated_rate(d)
        scale = 1.0 / sim
    else:
        d = econ.reprice(d, daily_to_annual(1.0), gross_margin, flat_penalty)  # 1.0/day
        scale = 1.0
    per = d.groupby(["model", "fold"])[["holding_cost", "stockout_cost"]].sum()
    mean = per.groupby("model").mean()
    return pd.DataFrame(
        {"holding_per_rate": mean["holding_cost"] * scale, "stockout": mean["stockout_cost"]}
    )


def cost_at(comp: pd.DataFrame, daily_rate: float) -> pd.Series:
    """Mean cost per fold for every model at `daily_rate`."""
    return comp["holding_per_rate"] * daily_rate + comp["stockout"]


def breakeven(comp: pd.DataFrame, a: str = MODEL_A, b: str = MODEL_B) -> float | None:
    """Daily rate at which models `a` and `b` cost the same, or None if their lines never cross at
    a positive rate (one model is cheaper at every rate)."""
    ds = comp.loc[a, "stockout"] - comp.loc[b, "stockout"]
    dh = comp.loc[b, "holding_per_rate"] - comp.loc[a, "holding_per_rate"]
    if dh == 0 or ds / dh <= 0:
        return None
    return float(ds / dh)


def cheaper_below(comp: pd.DataFrame, a: str, b: str) -> str | None:
    """Which of a/b is cheaper at rates *below* the crossover (the low-holding-cost side)."""
    if breakeven(comp, a, b) is None:
        return None
    return a if comp.loc[a, "stockout"] < comp.loc[b, "stockout"] else b


def pairwise_breakevens(comp: pd.DataFrame, ref: str = MODEL_A) -> pd.DataFrame:
    """`ref` against every other model: where the lines cross, and who wins on each side."""
    rows = []
    for other in comp.index.drop(ref):
        r = breakeven(comp, ref, other)
        if r is None:
            better = ref if cost_at(comp, 0.0)[ref] < cost_at(comp, 0.0)[other] else other
            rows.append({"other": other, "annual": None, "always_cheaper": better})
        else:
            rows.append({"other": other, "annual": daily_to_annual(r), "always_cheaper": None})
    return pd.DataFrame(rows)


def solve_numerically(
    detail: pd.DataFrame, a: str = MODEL_A, b: str = MODEL_B, tol: float = 1e-13
) -> float:
    """Bisection on the re-priced raw per-series rows — shares no code with `breakeven`, which
    works from aggregated components. If the two agree, the algebra and the aggregation are both
    right."""
    d = _csl(detail)
    sim = simulated_rate(d)

    def gap(rate: float) -> float:
        total = d["holding_cost"] * (rate / sim) + d["stockout_cost"]
        m = total.groupby([d["model"], d["fold"]]).sum().groupby("model").mean()
        return float(m[a] - m[b])

    lo, hi = 1e-9, 1.0
    if gap(lo) * gap(hi) > 0:
        raise ValueError(f"{a} and {b} do not cross between {lo} and {hi}/day")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(lo) * gap(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


@dataclass(frozen=True)
class BootstrapBreakeven:
    point: float  # annual
    ci_lo: float | None  # annual, over resamples where a positive crossover exists
    ci_hi: float | None
    frac_undefined: float  # resamples with no positive crossover
    a_cheaper_at: dict[float, float]  # annual rate -> share of resamples where `a` is cheaper


def bootstrap_breakeven(
    detail: pd.DataFrame,
    a: str = MODEL_A,
    b: str = MODEL_B,
    annual_rates: tuple[float, ...] = (),
    n_boot: int = boot.N_BOOT,
    seed: int = boot.SEED,
    alpha: float = 0.05,
) -> BootstrapBreakeven:
    """Paired bootstrap over series (same resample for both models, as everywhere in the project)
    of the crossover rate, and of "is `a` cheaper at rate r" for each rate in `annual_rates`.

    The crossover is a ratio of two differences, so it is unbounded when the holding difference
    shrinks toward zero; resamples with no positive crossover are counted, not dropped silently.
    """
    d = _csl(detail)
    sim = simulated_rate(d)
    da, db = d[d["model"] == a], d[d["model"] == b]
    ids = np.array(sorted(set(da["series_id"]) | set(db["series_id"])))
    keys = boot._group_keys(da, db, ["fold"])

    def mats(col: str) -> tuple[np.ndarray, np.ndarray]:
        return tuple(  # type: ignore[return-value]
            boot._matrix(x, col, "series_id", ["fold"], ids, keys) for x in (da, db)
        )

    h_a, h_b = mats("holding_cost")
    s_a, s_b = mats("stockout_cost")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))

    def resampled(m: np.ndarray) -> np.ndarray:
        return m[idx].sum(axis=1).mean(axis=1)

    hpr_a, hpr_b = resampled(h_a) / sim, resampled(h_b) / sim
    st_a, st_b = resampled(s_a), resampled(s_b)
    ds, dh = st_a - st_b, hpr_b - hpr_a
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dh != 0, ds / dh, np.nan)
    ok = np.isfinite(ratio) & (ratio > 0)
    annual = ratio[ok] * 365

    point_ds = float(s_a.sum(axis=0).mean() - s_b.sum(axis=0).mean())
    point_dh = float((h_b.sum(axis=0).mean() - h_a.sum(axis=0).mean()) / sim)
    lo, hi = (
        (float(v) for v in np.quantile(annual, [alpha / 2, 1 - alpha / 2]))
        if ok.any()
        else (
            None,
            None,
        )
    )
    cheaper = {
        r: float(np.mean((hpr_a - hpr_b) * annual_to_daily(r) + (st_a - st_b) < 0))
        for r in annual_rates
    }
    return BootstrapBreakeven(
        point=daily_to_annual(point_ds / point_dh),
        ci_lo=lo,
        ci_hi=hi,
        frac_undefined=float(1 - ok.mean()),
        a_cheaper_at=cheaper,
    )


def penalty_for_breakeven(
    comp: pd.DataFrame, target_annual: float, penalty: float, a: str = MODEL_A, b: str = MODEL_B
) -> float:
    """Stockout penalty at which the crossover would sit at `target_annual`.

    Stockout cost is `penalty x units lost`, so the crossover scales linearly with the penalty —
    the rate is one side of a ratio, and the penalty (also an unsourced Track A default) is the
    other."""
    r = breakeven(comp, a, b)
    if r is None:
        raise ValueError("no crossover")
    return penalty * annual_to_daily(target_annual) / r


def curve_frame(comp: pd.DataFrame, annual_rates: np.ndarray) -> pd.DataFrame:
    """Long frame (annual_rate, model, cost) for plotting."""
    rows = [
        {"annual_rate": r, "model": m, "cost": c}
        for r in annual_rates
        for m, c in cost_at(comp, annual_to_daily(r)).items()
    ]
    return pd.DataFrame(rows)


# ---- chart -----------------------------------------------------------------------------------

# dataviz reference palette, light mode (categorical slot 1 blue, slot 2 orange; validated with
# scripts/validate_palette.js: adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e8e7e3"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BAND = "#e87ba4"  # never a series colour; the shading carries a label, not identity


def plot_cost_vs_rate(
    comp: pd.DataFrame,
    path,
    a: str = MODEL_A,
    b: str = MODEL_B,
    legacy_daily: float = LEGACY_SIMULATED_RATE,
    x_max_pct: float = 800.0,
    comp_opt: pd.DataFrame | None = None,
    opt_label: str = "cost-optimal target",
) -> None:
    """Both models' cost as a function of the holding rate (top), and the gap between them (bottom),
    with the crossover marked and the plausible range shaded. `comp_opt`, if given, adds the gap at
    the cost-optimal service target as a dashed line: the crossover depends on the target too.

    Linear x-axis on purpose: cost is exactly linear in the rate, so the curves are the straight
    lines they are, and the distance between the shaded band and the crossover is drawn at its true
    size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    be = daily_to_annual(breakeven(comp, a, b) or np.nan) * 100  # percent per year
    lo, hi = (v * 100 for v in PLAUSIBLE_ANNUAL_HOLDING_RATE)
    legacy = daily_to_annual(legacy_daily) * 100
    xs = np.linspace(0, x_max_pct, 400)
    cost = curve_frame(comp, xs / 100).pivot(index="annual_rate", columns="model", values="cost")
    gap = cost[a].to_numpy() - cost[b].to_numpy()
    y_at_be = float(cost_at(comp, annual_to_daily(be / 100))[a])
    ymax = float(cost[[a, b]].to_numpy().max()) * 1.18
    gap_opt = None
    if comp_opt is not None:
        co = curve_frame(comp_opt, xs / 100).pivot(
            index="annual_rate", columns="model", values="cost"
        )
        gap_opt = co[a].to_numpy() - co[b].to_numpy()
    gmax = float(max(np.abs(gap).max(), 0 if gap_opt is None else np.abs(gap_opt).max())) * 1.3

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 6.6),
        dpi=160,
        sharex=True,
        facecolor=SURFACE,
        gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.1},
    )
    for axis in (ax, ax2):
        axis.set_facecolor(SURFACE)
        axis.axvspan(lo, hi, color=BAND, alpha=0.3, lw=0)
        axis.axvline(be, color=INK_2, lw=1, ls=(0, (1, 3)), zorder=1)
        axis.axvline(legacy, color=INK_2, lw=1, ls=(0, (1, 3)), zorder=1)
        axis.grid(axis="y", color=GRID, lw=1)
        axis.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)

    for m, col in ((a, BLUE), (b, ORANGE)):
        ax.plot(xs, cost[m].to_numpy(), color=col, lw=2, solid_capstyle="round", label=m, zorder=3)
    ax.plot([be], [y_at_be], "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=4)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Mean cost per 28-day fold (95% target)", color=INK_2)
    ax.legend(
        loc="upper left", frameon=False, labelcolor=INK, fontsize=10, bbox_to_anchor=(0.09, 1)
    )
    ax.annotate(
        "plausible carrying cost\n15–30%/yr",
        xy=((lo + hi) / 2, ymax * 0.03),
        xytext=(60, ymax * 0.08),
        color=INK,
        fontsize=9.5,
        va="center",
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8},
    )
    ax.annotate(
        f"crossover {be:,.0f}%/yr\n{be / hi:.1f}× the top of the plausible range",
        xy=(be, y_at_be),
        xytext=(be - 30, y_at_be * 2.3),
        color=INK,
        fontsize=9.5,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8},
    )
    ax.text(
        legacy - 12,
        ymax * 0.1,
        "original default\n730%/yr (2%/day)",
        ha="right",
        va="center",
        color=INK_2,
        fontsize=9.5,
    )

    ax2.plot(xs, gap, color=INK, lw=2, solid_capstyle="round", zorder=3, label="95% target")
    ax2.axhline(0, color=INK_2, lw=1, zorder=2)
    ax2.plot([be], [0], "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=4)
    if gap_opt is not None:
        ax2.plot(xs, gap_opt, color=INK, lw=2, ls=(0, (4, 3)), zorder=3, label=opt_label)
        be_opt = breakeven(comp_opt, a, b)
        if be_opt is not None:
            x_opt = daily_to_annual(be_opt) * 100
            ax2.plot([x_opt], [0], "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=4)
            ax2.annotate(
                f"{x_opt:,.0f}%/yr",
                xy=(x_opt, 0),
                xytext=(x_opt + 25, -gmax * 0.55),
                color=INK,
                fontsize=9.5,
                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8},
            )
    ax2.set_ylim(-gmax, gmax)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}" if v else "0"))
    ax2.set_ylabel(f"{a} − {b}", color=INK_2)
    ax2.text(x_max_pct * 0.3, gmax * 0.55, f"{b} cheaper", ha="center", color=INK, fontsize=9.5)
    ax2.text(x_max_pct * 0.62, -gmax * 0.72, f"{a} cheaper", ha="center", color=INK, fontsize=9.5)
    ax2.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=9)
    ax2.set_xlim(0, x_max_pct)
    ax2.xaxis.set_major_locator(MultipleLocator(100))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax2.set_xlabel("Holding cost, % of unit cost per year", color=INK_2)

    fig.text(
        0.012,
        0.965,
        f"{b} is cheaper at every plausible holding cost (at the default 95% target)",
        color=INK,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.012,
        0.925,
        (
            f"{a} wins only above ~{be:,.0f}%/yr; at a cost-optimal target the gap "
            "all but vanishes."
            if comp_opt is not None
            else f"{a} wins only above ~{be:,.0f}%/yr."
        ),
        color=INK_2,
        fontsize=10,
        va="top",
    )
    fig.subplots_adjust(top=0.88, left=0.12, right=0.97, bottom=0.09)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- report ----------------------------------------------------------------------------------


def resimulation_check(
    rate_annual: float = DEFAULT_ANNUAL_HOLDING_RATE, gross_margin: float | None = None
) -> dict:
    """Re-run the shipped safety-stock policy under (`rate_annual`, `gross_margin`) through the
    real simulation code, from the cached forecasts, and compare with the stored legacy run
    re-priced by `economics.reprice`.

    Tests the two premises every number here rests on: the policy does not react to what holding or
    a lost sale costs (trajectory identical), and re-pricing — including the per-SKU price weights
    of unit economics — reproduces a genuine re-simulation exactly."""
    from dataclasses import replace

    from reorderpoint import ablate_safety_stock as ab
    from reorderpoint import backtest as bt
    from reorderpoint import divergence as dv
    from reorderpoint.config import load_config

    costs = replace(
        load_config().costs,
        holding_cost_rate=annual_to_daily(rate_annual),
        gross_margin=gross_margin,
    )
    panel = bt.load_eval_panel()
    forecasts = pd.read_parquet(dv.FORECASTS_PATH)
    calib = ab.calibration_residuals(panel, costs)
    _, new, _ = ab.run_cell(panel, forecasts, calib, *ab.BASELINE_CELL, costs, n_boot=20)
    old = pd.read_parquet(ab.CELL_DETAIL_PATH)
    old = old[(old["form"] == ab.BASELINE_CELL[0]) & (old["granularity"] == ab.BASELINE_CELL[1])]
    old = econ.reprice(old, rate_annual, gross_margin)
    m = old.merge(new, on=["model", "fold", "series_id"], suffixes=("_old", "_new"))
    trajectory = ["units_shipped", "orders", "avg_on_hand", "cycles", "stockout_cycles"]
    priced_cols = ["holding_cost", "stockout_cost", "total_cost"]
    return {
        "rows": len(m),
        "trajectory_identical": all(
            np.allclose(m[f"{c}_old"], m[f"{c}_new"], rtol=1e-9, atol=1e-9) for c in trajectory
        ),
        "repriced_max_abs_diff": float(
            max((m[f"{c}_old"] - m[f"{c}_new"]).abs().max() for c in priced_cols)
        ),
        "rate_annual": rate_annual,
        "gross_margin": gross_margin,
    }


def _usd(v: float) -> str:
    return f"${v:,.0f}"


def _gap_ci(priced: pd.DataFrame, target: float | None = None) -> str:
    d = priced if target is None else priced[np.abs(priced["target"] - target) < 1e-9]
    d = _csl(d)
    out = boot.bootstrap_diff(
        d[d["model"] == MODEL_A], d[d["model"] == MODEL_B], "total_cost", group_cols=["fold"]
    )
    span = f"[{out['ci_lo']:+,.0f}, {out['ci_hi']:+,.0f}]"
    return f"{out['point']:+,.0f} {span}" + (" — crosses zero" if out["crosses_zero"] else "")


def build_report(
    comp: pd.DataFrame,
    comp_legacy: pd.DataFrame,
    comp_opt: pd.DataFrame | None,
    opt_target: float | None,
    detail: pd.DataFrame,
    bb: BootstrapBreakeven,
    numeric: float,
    resim: dict | None,
    costs,
    mean_price: float | None,
    chart_rel: str,
    gap_cis: dict[str, str],
) -> tuple[str, dict]:
    from reorderpoint import backtest as bt

    a, b = MODEL_A, MODEL_B
    m0 = costs.gross_margin
    r_star = breakeven(comp, a, b)
    assert r_star is not None
    be = daily_to_annual(r_star) * 100  # percent per year
    top = PLAUSIBLE_ANNUAL_HOLDING_RATE[1]
    lo_pl = PLAUSIBLE_ANNUAL_HOLDING_RATE[0]
    legacy_be = daily_to_annual(breakeven(comp_legacy, a, b)) * 100
    opt_be = None
    if comp_opt is not None and breakeven(comp_opt, a, b) is not None:
        opt_be = daily_to_annual(breakeven(comp_opt, a, b)) * 100
    numeric_err = abs(numeric - r_star) / r_star

    def price_row(label: str, c: pd.DataFrame, r: float) -> dict:
        cs = cost_at(c, annual_to_daily(r))
        return {
            "holding rate": label,
            a: _usd(cs[a]),
            b: _usd(cs[b]),
            f"{a} − {b}": f"{cs[a] - cs[b]:+,.0f}",
            "cheapest of five": cs.idxmin(),
        }

    at_rows = pd.DataFrame(
        [price_row(f"{r:.0%}/yr", comp, r) for r in (0.15, 0.20, 0.25, 0.30, 1.0)]
        + [price_row(f"{be:,.0f}%/yr (crossover)", comp, be / 100)]
        + [price_row("730%/yr (the original 2%/day)", comp, daily_to_annual(0.02))]
    )
    at_rows.loc[at_rows["holding rate"].str.contains("crossover"), "cheapest of five"] = (
        f"tie: {a} = {b}"
    )

    three = [
        {
            "cost model": f"unit economics ({m0:.1%} margin) — default",
            "service target": "95% (default)",
            "crossover": f"{be:,.0f}%/yr",
            f"{a} − {b} at 25%/yr": gap_cis["econ95"],
        },
        {
            "cost model": f"unit economics ({m0:.1%} margin) — default",
            "service target": f"{opt_target:.4%} (cost-optimal)" if opt_target else "—",
            "crossover": "—" if opt_be is None else f"{opt_be:,.0f}%/yr",
            f"{a} − {b} at 25%/yr": gap_cis.get("econ_opt", "—"),
        },
        {
            "cost model": "legacy flat $5.00/unit, stock at price",
            "service target": "95% (default)",
            "crossover": f"{legacy_be:,.0f}%/yr",
            f"{a} − {b} at 25%/yr": gap_cis["legacy95"],
        },
    ]

    pw_rows = []
    for other in comp.index.drop(a):
        r = breakeven(comp, a, other)
        if r is None:
            dominant = a if cost_at(comp, 0.0)[a] < cost_at(comp, 0.0)[other] else other
            pw_rows.append(
                {f"{a} vs.": other, "crossover": "none", "cheaper": f"{dominant}, always"}
            )
        else:
            low = cheaper_below(comp, a, other)
            high = a if low == other else other
            pw_rows.append(
                {
                    f"{a} vs.": other,
                    "crossover": f"{daily_to_annual(r):,.0%}/yr",
                    "cheaper": f"{low} below it, {high} above",
                }
            )

    ci = (
        f"[{bb.ci_lo:,.0%}, {bb.ci_hi:,.0%}]"
        if bb.ci_lo is not None
        else "not estimable (no positive crossover in any resample)"
    )
    p_rows = pd.DataFrame(
        [
            {"holding rate": f"{r:.0%}/yr", f"share of resamples where {a} is cheaper": f"{p:.1%}"}
            for r, p in bb.a_cheaper_at.items()
        ]
    )
    # the lost-sale margin at which the crossover would fall to the top of the plausible range:
    # h*(m) = k * m/(1-m), k fixed by the default margin
    k = (be / 100) / (m0 / (1 - m0))
    ratio = top / k
    m_needed = ratio / (1 + ratio)

    lines = [
        f"# Holding-cost breakeven — {dt.date.today().isoformat()}",
        "",
        f"**Under the default cost parameters — holding as a share of unit cost per year, a lost "
        f"sale costing {m0:.1%} of price (Walmart's gross margin) — {a} is the cheapest model only "
        f"if holding a unit for a year costs more than {be / 100:.1f}× its unit cost "
        f"({be:,.0f}%/yr), {be / (top * 100):.1f}× the top of the plausible "
        f"{lo_pl:.0%}–{top:.0%}/yr range. At any carrying cost in that range {b} is cheaper — "
        f"and the cheapest of the five models — at the default 95% service target.** Two "
        f"conditions come with that, and are part of the finding: at the cost-optimal service "
        + (
            f"target ({opt_target:.4%}) the crossover falls to {opt_be:,.0f}%/yr and the gap is "
            "not statistically distinguishable from zero "
            if opt_be is not None
            else "target the crossover falls to the edge of the range "
        )
        + f"(`reports/optimal_target_*.md`); and under the legacy flat-$5 penalty it was "
        f"{legacy_be:,.0f}%/yr (`reports/penalty_sensitivity_*.md`). The original Phase 4 "
        "result had LightGBM cheapest at a holding rate (2%/*day*, 730%/yr) no business pays.",
        "",
        f"![Cost vs. holding rate: both models' curves, the crossover, and the plausible range]"
        f"({chart_rel})",
        "",
        "## 1. The breakeven",
        "",
        "Mean simulated cost per 28-day fold, shipped policy (CSL 95%, normal × intermittency), "
        "400 series × 4 folds. Unit economics: a lost sale costs the unit's gross margin and "
        "stock is valued at cost, `(1 − margin) × price` (`docs/decision.md`). The stored runs "
        "are re-priced exactly per series (`economics.reprice`; see §3).",
        "",
        bt._markdown_table(at_rows, float_cols=()),
        "",
        "Holding and stockout cost per fold, and the cost function they imply "
        "(`cost = holding_per_rate × daily_rate + stockout`):",
        "",
        bt._markdown_table(
            comp.assign(
                holding_per_rate=comp["holding_per_rate"].map("{:,.0f}".format),
                stockout=comp["stockout"].map("{:,.1f}".format),
            )
            .reset_index()
            .rename(
                columns={
                    "holding_per_rate": "holding per 1.0/day of rate",
                    "stockout": "stockout (rate-independent)",
                }
            ),
            float_cols=(),
        ),
        "",
        f"Solving `cost_{a}(r) = cost_{b}(r)`: `r* = (S_{a} − S_{b}) / (H_{b} − H_{a})` = "
        f"({comp.loc[a, 'stockout']:,.1f} − {comp.loc[b, 'stockout']:,.1f}) / "
        f"({comp.loc[b, 'holding_per_rate']:,.0f} − {comp.loc[a, 'holding_per_rate']:,.0f}) = "
        f"**{r_star:.6f}/day = {be:,.2f}%/yr**. The two models trade one cost for the other: "
        f"{a} carries less stock and stocks out more. Carrying less pays only when carrying is "
        "expensive; below the crossover the extra stockouts cost more than the stock saved.",
        "",
        "## 2. Three parameters decide the ranking",
        "",
        "The crossover is a property of the cost model *and* the service target, not of the "
        "holding rate alone:",
        "",
        bt._markdown_table(pd.DataFrame(three), float_cols=()),
        "",
        "## 3. Verified, not fitted",
        "",
        f"1. **Closed form** from the aggregated components: {be:,.4f}%/yr.",
        f"2. **Independent root-find** — bisection on the re-priced *raw per-series rows*, sharing "
        f"no code with (1): {daily_to_annual(numeric) * 100:,.4f}%/yr (relative difference "
        f"{numeric_err:.1e}).",
    ]
    if resim is not None:
        lines += [
            f"3. **Re-simulation at {resim['rate_annual']:.0%}/yr"
            + (
                ", legacy flat penalty"
                if resim["gross_margin"] is None
                else f", {resim['gross_margin']:.1%} gross margin"
            )
            + f"** through the real simulation code from the cached forecasts "
            f"({resim['rows']:,} series × fold × model rows): stockouts, shipments, orders, cycles "
            f"and average on-hand are "
            f"{'**identical**' if resim['trajectory_identical'] else '**NOT identical**'} to the "
            f"stored run, and every priced column (holding, stockout, total) equals the stored "
            f"run re-priced to within {resim['repriced_max_abs_diff']:.1e}. This tests both "
            "premises the analysis rests on: the policy does not react to what holding or a lost "
            "sale costs, and re-pricing — including unit economics' per-SKU price weights — "
            "reproduces a genuine re-simulation.",
        ]
    else:
        lines += ["3. Re-simulation: skipped (forecast cache not on disk)."]
    lines += [
        "",
        "## 4. Sampling uncertainty on the crossover",
        "",
        "Paired bootstrap over the 400 series (2,000 resamples, one draw shared by both models), "
        "as in `model_ci.py`, default economics, 95% target.",
        "",
        f"Crossover: **{be:,.0f}%/yr, 95% CI {ci}**"
        + (
            f"; {bb.frac_undefined:.1%} of resamples had no positive crossover."
            if bb.frac_undefined
            else "."
        ),
        "",
        bt._markdown_table(p_rows, float_cols=()),
        "",
        "## 5. Against the other models",
        "",
        bt._markdown_table(pd.DataFrame(pw_rows), float_cols=()),
        "",
        "## 6. What this depends on",
        "",
        f"- **The lost-sale margin** ({m0:.1%}, Walmart U.S. segment FY2026 gross profit rate — a "
        "lower bound on what a stockout costs). The crossover is proportional to "
        f"`margin / (1 − margin)`: it would reach the top of the plausible holding range "
        f"({top:.0%}/yr) at a margin of {m_needed:.1%}. Full sweep: "
        "`reports/penalty_sensitivity_*.md`."
        + (f" Mean unit price ${mean_price:.2f}." if mean_price else ""),
        "- **The service target.** The 95% default was never a finding; the cost-optimal nominal "
        "target is far higher and the ranking there is different (`reports/optimal_target_*.md`).",
        "- **The plausible range** is the project's working range. The only bound with a named "
        'source is 25%/yr (`docs/decision.md`, "Holding cost rate"); the reading of the result '
        f"does not depend on the exact bounds — at the crossover a unit held for a year costs "
        f"{be / 100:.1f}× the unit itself.",
        "- **A business's own figures** replace the assumptions; rerun `make breakeven` and "
        "the chart, the README figure and the dashboard sliders all follow.",
        "",
    ]
    return "\n".join(lines), {"annual": be, "ci": (bb.ci_lo, bb.ci_hi), "opt_annual": opt_be}


def main() -> None:
    from reorderpoint import backtest as bt
    from reorderpoint import decision as dec
    from reorderpoint import optimal_target as ot
    from reorderpoint.config import load_config

    costs = load_config().costs
    gm = costs.gross_margin
    detail = pd.read_parquet(dec.DETAIL_PATH)
    comp = model_components(detail, gross_margin=gm)
    comp_legacy = model_components(detail, gross_margin=None)
    r_star = breakeven(comp)
    if r_star is None:
        raise SystemExit(f"{MODEL_A} and {MODEL_B} never cross; nothing to report")

    priced = econ.reprice(detail, DEFAULT_ANNUAL_HOLDING_RATE, gm)
    numeric = solve_numerically(priced)
    if abs(numeric - r_star) / r_star > 1e-6:
        raise SystemExit(f"closed form {r_star} and numeric root {numeric} disagree")

    bb = bootstrap_breakeven(priced, annual_rates=(0.15, 0.20, 0.25, 0.30))
    gap_cis = {
        "econ95": _gap_ci(priced),
        "legacy95": _gap_ci(econ.reprice(detail, DEFAULT_ANNUAL_HOLDING_RATE, None)),
    }

    # the same comparison at the cost-optimal service target, from the extended sweep
    comp_opt = opt_target = None
    if ot.DETAIL_PATH.exists():
        sweep = pd.read_parquet(ot.DETAIL_PATH)
        p_sweep = ot.priced(sweep, DEFAULT_ANNUAL_HOLDING_RATE, gm)
        opt_target = ot.argmin_target(ot.curves(p_sweep))["target"]
        d_opt = sweep[ot.at_target(sweep, opt_target)]
        comp_opt = model_components(d_opt, gross_margin=gm)
        gap_cis["econ_opt"] = _gap_ci(p_sweep, opt_target)

    try:
        resim = resimulation_check(gross_margin=gm)
    except (FileNotFoundError, OSError):
        resim = None
    if resim is not None and not (
        resim["trajectory_identical"] and resim["repriced_max_abs_diff"] < 1e-6
    ):
        raise SystemExit(f"re-simulation disagrees with re-pricing: {resim}")

    mean_price = None
    try:
        panel = bt.load_eval_panel()
        train = panel[panel["date"] <= bt.make_folds(panel)[-1].train_end]
        mean_price = float(dec.unit_cost_from_price(train).mean())
    except (FileNotFoundError, OSError):
        pass

    files = bt.REPORTS_DIR / "holding_breakeven_files"
    files.mkdir(parents=True, exist_ok=True)
    chart = files / "cost_vs_holding_rate.png"
    plot_cost_vs_rate(
        comp,
        chart,
        comp_opt=comp_opt,
        opt_label=f"{opt_target:.4%} target (cost-optimal)" if opt_target else "",
    )

    report, stats = build_report(
        comp,
        comp_legacy,
        comp_opt,
        opt_target,
        detail,
        bb,
        numeric,
        resim,
        costs,
        mean_price,
        f"holding_breakeven_files/{chart.name}",
        gap_cis,
    )
    out = bt.REPORTS_DIR / f"holding_breakeven_{dt.date.today().isoformat()}.md"
    out.write_text(report)
    print(f"Wrote {out} and {chart}")
    print(stats)


if __name__ == "__main__":
    main()
