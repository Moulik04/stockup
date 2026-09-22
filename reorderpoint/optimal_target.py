"""What service target do the costs actually call for, and does the model ranking survive it?

Every model ranking in this project was measured at a nominal target of 95%. That number was a
default, not a finding. With holding at a real carrying cost the newsvendor critical ratio
`CR = Cu / (Cu + Co)` — the cycle service level that minimises expected cost — is ~98-99%, so the
cost-optimal *nominal* target is well above 95% and the sweep in `service_level_sweep.py`, which
stopped at 99.9%, ran into its own grid edge. Two questions follow, answered here:

1. **Where is the optimum?** The grid is extended to 99.9999% (`z` = 4.75). The shipped scheme is
   simulated at every level for every model, keeping the per-series rows so any (holding rate,
   gross margin) can be applied exactly (`economics.reprice`) — trajectories depend on the target
   and the forecast only. The report says whether an interior optimum now exists, what realised
   cycle service level each nominal target buys, and how that compares with the critical ratio.
2. **Does the ranking change there?** LightGBM vs SeasonalNaive vs the statistical baselines are
   compared at 95% and at the cost-optimal target with paired-bootstrap CIs over series. Tail
   behaviour dominates at high targets, so the 95% ranking need not hold.

Cost model: unit economics by default (`config.DEFAULT_GROSS_MARGIN`, holding at 25%/yr on stock
valued at cost); the legacy flat-$5 pricing is reported beside it for comparability.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import norm

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import divergence as dv
from reorderpoint import economics as econ
from reorderpoint import service_level_sweep as sw
from reorderpoint.config import (
    DEFAULT_ANNUAL_HOLDING_RATE,
    CostParams,
    annual_to_daily,
    load_config,
)

EXTRA_LEVELS = [0.9995, 0.9999, 0.99995, 0.99999, 0.999999]
LEVELS = sw.LEVELS + EXTRA_LEVELS
DETAIL_PATH = bt.PANEL_PATH.parent / "target_sweep_detail.parquet"
MODELS = ["LightGBM", "SeasonalNaive", "AutoETS", "AutoTheta", "MovingAverage"]
FOCUS = ("LightGBM", "SeasonalNaive")
REF_TARGET = 0.95


def run_sweep(costs: CostParams, levels: list[float] = LEVELS) -> pd.DataFrame:
    """The shipped scheme, every model, every level: per-series rows tagged with the target.

    Cached: levels already on disk are not re-simulated. Rows are stamped with the pricing they
    were simulated under, so they can be re-priced to anything (`economics.reprice`)."""
    have = pd.read_parquet(DETAIL_PATH) if DETAIL_PATH.exists() else None
    done = set() if have is None else set(np.round(have["target"].unique(), 8))
    todo = [lv for lv in levels if round(lv, 8) not in done]
    frames = [] if have is None else [have]
    if todo:
        panel = bt.load_eval_panel()
        forecasts = pd.read_parquet(dv.FORECASTS_PATH)
        calib = ab.calibration_residuals(panel, costs)
        for level in todo:
            _, detail, _ = ab.run_cell(
                panel, forecasts, calib, *ab.BASELINE_CELL, costs, n_boot=20, service_level=level
            )
            frames.append(detail.assign(target=level))
            print(f"target sweep: {level:.6f}")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DETAIL_PATH, index=False)
    return out


def at_target(df: pd.DataFrame, target: float) -> pd.Series:
    """Boolean mask for rows at `target`. Not `np.isclose`: its default tolerance (rtol 1e-5)
    treats 99.999% and 99.9999% as the same target, which on this grid silently selects both."""
    return np.abs(df["target"] - target) < 1e-9


def priced(detail: pd.DataFrame, annual_holding: float, gross_margin: float | None):
    return econ.reprice(detail, annual_holding, gross_margin)


def curves(p: pd.DataFrame) -> pd.DataFrame:
    """Per (target, model): mean over folds of the fold's summed cost, plus pooled realised CSL
    and fill rate (pooled over series within a fold, then averaged, as everywhere else)."""
    g = p.groupby(["target", "model", "fold"]).agg(
        cost=("total_cost", "sum"),
        holding=("holding_cost", "sum"),
        stockout=("stockout_cost", "sum"),
        cycles=("cycles", "sum"),
        bad=("stockout_cycles", "sum"),
        shipped=("units_shipped", "sum"),
        demanded=("units_demanded", "sum"),
    )
    g["csl"] = 1 - g["bad"] / g["cycles"]
    g["fill"] = g["shipped"] / g["demanded"]
    return (
        g.reset_index()
        .groupby(["target", "model"])[["cost", "holding", "stockout", "csl", "fill"]]
        .mean()
        .reset_index()
    )


def pooled(c: pd.DataFrame) -> pd.DataFrame:
    """Mean over the five models — the quantity the earlier sweeps optimised."""
    return c.groupby("target")[["cost", "holding", "stockout", "csl", "fill"]].mean().reset_index()


def argmin_target(c: pd.DataFrame, model: str | None = None) -> dict:
    d = pooled(c) if model is None else c[c["model"] == model]
    d = d.sort_values("target")
    best = d.loc[d["cost"].idxmin()]
    return {
        "target": float(best["target"]),
        "cost": float(best["cost"]),
        "csl": float(best["csl"]),
        "fill": float(best["fill"]),
        "at_edge": bool(best["target"] >= d["target"].max() - 1e-12),
    }


def critical_ratio(costs: CostParams, price, period_days: float):
    """Cu / (Cu + Co) with Cu the per-unit lost-sale cost and Co = rate x unit cost x period."""
    cu = costs.stockout_penalty(price)
    co = costs.holding_cost_rate * costs.unit_cost(price) * period_days
    return cu / (cu + co)


def implied_targets(
    costs: CostParams, prices: pd.Series, cycle_days: float, annual_holding: float
) -> pd.DataFrame:
    """The critical ratio — the optimal realised CSL — under each pricing, for the two protection
    periods the project has used (the lead time, and the measured mean replenishment cycle)."""
    rows = []
    variants = {
        "unit economics (lost margin)": costs,
        f"legacy flat ${costs.stockout_penalty_per_unit:.2f}/unit": costs.flat(),
    }
    for label, c in variants.items():
        c = CostParams(
            annual_to_daily(annual_holding),
            c.stockout_penalty_per_unit,
            c.lead_time_days,
            c.service_level_target,
            c.gross_margin,
        )
        for period_label, period in (
            (f"lead time ({c.lead_time_days}d)", c.lead_time_days),
            (f"measured cycle ({cycle_days:.1f}d)", cycle_days),
        ):
            per = np.asarray(critical_ratio(c, prices.to_numpy(), period), dtype=float)
            rows.append(
                {
                    "pricing": label,
                    "protection period": period_label,
                    "CR at mean price": float(critical_ratio(c, float(prices.mean()), period)),
                    "CR p10": float(np.quantile(per, 0.1)),
                    "CR median": float(np.median(per)),
                    "CR p90": float(np.quantile(per, 0.9)),
                }
            )
    return pd.DataFrame(rows)


def nominal_for_csl(c: pd.DataFrame, cr: float) -> float | None:
    """Smallest grid target whose pooled realised CSL reaches `cr`; None if none does."""
    d = pooled(c).sort_values("target")
    ok = d[d["csl"] >= cr]
    return float(ok["target"].iloc[0]) if len(ok) else None


def flat_from(c: pd.DataFrame, tol: float = 0.02) -> float:
    """Smallest grid target whose pooled cost is within `tol` of the minimum — where the curve has
    effectively stopped falling, which is the honest statement when the argmin sits at the grid's
    edge or moves with the cost parameters."""
    d = pooled(c).sort_values("target")
    return float(d[d["cost"] <= d["cost"].min() * (1 + tol)]["target"].iloc[0])


def compare_at(p: pd.DataFrame, target: float) -> pd.DataFrame:
    """LightGBM minus each other model at `target`: point, paired-bootstrap CI over series."""
    at = p[at_target(p, target)]
    a = at[at["model"] == "LightGBM"]
    rows = []
    for other in MODELS[1:]:
        out = boot.bootstrap_diff(a, at[at["model"] == other], "total_cost", group_cols=["fold"])
        rows.append({"other": other, **out})
    return pd.DataFrame(rows)


def ranking_table(c: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    rows = []
    for t in targets:
        d = c[at_target(c, t)].set_index("model")["cost"]
        ranks = d.rank().astype(int)
        for m in MODELS:
            rows.append({"target": t, "model": m, "cost": d[m], "rank": ranks[m]})
    return pd.DataFrame(rows)


# ---- is it the sizing formula? ----------------------------------------------------------------

SATURATION_CELLS = [
    ("normal", "intermittency"),
    ("normal", "volume_quintile"),
    ("empirical", "intermittency"),
    ("empirical", "volume_tercile_within_intermittency"),
]
SATURATION_TARGETS = [0.99, 0.9999, 0.999999]


def scheme_saturation(
    costs: CostParams,
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    calib: pd.DataFrame,
    cells: list[tuple[str, str]] = SATURATION_CELLS,
    targets: list[float] = SATURATION_TARGETS,
) -> pd.DataFrame:
    """Realised CSL and fill rate of each safety-stock scheme at a few nominal targets, `S = s`.

    The question it answers: does a *better safety-stock formula* (finer pooling, or the empirical
    quantile instead of z x std) lift the ~97% ceiling on realised service? If every scheme
    saturates in the same narrow band, the formula is not the binding constraint. Pooled the way
    the rest of the project pools (within fold, then averaged over folds and models)."""
    rows = []
    for form, gran in cells:
        for t in targets:
            _, d, _ = ab.run_cell(
                panel, forecasts, calib, form, gran, costs, n_boot=20, service_level=t
            )
            g = d.groupby(["model", "fold"]).agg(
                cycles=("cycles", "sum"),
                bad=("stockout_cycles", "sum"),
                shipped=("units_shipped", "sum"),
                demanded=("units_demanded", "sum"),
            )
            rows.append(
                {
                    "form": form,
                    "granularity": gran,
                    "target": t,
                    "realised_csl": float((1 - g["bad"] / g["cycles"]).mean()),
                    "fill": float((g["shipped"] / g["demanded"]).mean()),
                }
            )
    return pd.DataFrame(rows)


# ---- why realised service saturates ----------------------------------------------------------


def cycle_outcomes(
    demand: np.ndarray,
    s: float,
    lead_time_days: int,
    S: float | None = None,
    on_hand_start: float | None = None,
) -> list[tuple[bool, bool]]:
    """(stocked out?, lead-time demand exceeded s?) for each replenishment cycle of an (s, S)
    policy over `demand` — `decision.simulate_series`'s logic, replicated so a stockout can be
    attributed rather than only counted. `S` defaults to `s` (the original policy) and the start
    stock to `s`.

    An order triggers when on-hand falls *below* s, so the stock held when it is placed is at most
    s. A stockout cycle where demand exceeded s is a sizing failure (the buffer was too small).
    One where demand stayed at or below s but exceeded the on-hand actually held is an
    **undershoot** — inventory was below s when it mattered. The two call for different fixes; a
    wider buffer only addresses the first. (Under `S = s` the undershoot is large because an order
    replaces only the deficit at the moment it is placed, so demand arriving during the lead time
    is not covered until the next order; a larger `S` refills past s.)"""
    S = s if S is None else S
    on_hand = s if on_hand_start is None else on_hand_start
    pending = None
    out: list[tuple[bool, bool]] = []
    open_, had, exposure = False, False, 0.0
    for day in range(len(demand)):
        if pending is not None and pending[0] == day:
            on_hand += pending[1]
            pending = None
            if open_:
                out.append((had, exposure > s + 1e-9))
                open_, had = False, False
        d = float(demand[day])
        shipped = min(on_hand, d)
        if open_:
            exposure += d
            if d - shipped > 0:
                had = True
        on_hand -= shipped
        if on_hand < s and pending is None and S - on_hand > 0:
            pending = (day + lead_time_days, S - on_hand)
            open_, had, exposure = True, False, 0.0
    return out


def stockout_decomposition(
    detail: pd.DataFrame, panel: pd.DataFrame, target: float | None, lead_time_days: int
) -> dict:
    """Split every stockout cycle into sizing failures and undershoot, across all models, folds and
    series in `detail` (filtered to `target` when given). Uses each row's `order_up_to` when the
    run stamped one. Verified against the stored simulation: the replicated cycle and
    stockout-cycle counts must equal the stored ones or this raises."""
    d = detail if target is None else detail[at_target(detail, target)]
    folds = {f.index: f for f in bt.make_folds(panel)}
    cycles = stockouts = sizing = 0
    has_S = "order_up_to" in d.columns
    for (fold, _model), g in d.groupby(["fold", "model"]):
        f = folds[fold]
        test = panel[(panel["date"] >= f.test_start) & (panel["date"] <= f.test_end)]
        demand = {
            sid: x.sort_values("date")["y"].to_numpy() for sid, x in test.groupby("series_id")
        }
        sid_col, s_col = g["series_id"].to_numpy(), g["reorder_point"].to_numpy(dtype=float)
        S_col = g["order_up_to"].to_numpy(dtype=float) if has_S else s_col
        start_col = (
            g["on_hand_start"].to_numpy(dtype=float) if "on_hand_start" in g.columns else s_col
        )
        for sid, s_i, S_i, start in zip(sid_col, s_col, S_col, start_col, strict=True):
            for had, exceeded in cycle_outcomes(demand[sid], s_i, lead_time_days, S_i, start):
                cycles += 1
                if had:
                    stockouts += 1
                    sizing += int(exceeded)
    if cycles != int(d["cycles"].sum()) or stockouts != int(d["stockout_cycles"].sum()):
        raise ValueError(
            f"replica disagrees with the stored simulation: {cycles}/{stockouts} vs "
            f"{int(d['cycles'].sum())}/{int(d['stockout_cycles'].sum())}"
        )
    return {
        "target": target,
        "cycles": cycles,
        "stockout_cycles": stockouts,
        "sizing_failures": sizing,
        "undershoot": stockouts - sizing,
        "undershoot_share": (stockouts - sizing) / stockouts if stockouts else float("nan"),
        "csl_if_sizing_only": 1 - sizing / cycles,
    }


# ---- chart -----------------------------------------------------------------------------------

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
# dataviz reference palette, categorical slots 1-5 in fixed order (validated adjacent pairs).
COLORS = {
    "LightGBM": "#2a78d6",
    "SeasonalNaive": "#eb6834",
    "AutoETS": "#1baf7a",
    "AutoTheta": "#eda100",
    "MovingAverage": "#e87ba4",
}


def _nines(t: np.ndarray | float):
    return -np.log10(1 - np.asarray(t, dtype=float))


def plot_target_curves(c: pd.DataFrame, cr: float | None, opt: float, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 6.8),
        dpi=160,
        sharex=True,
        facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.1},
    )
    for axis in (ax, ax2):
        axis.set_facecolor(SURFACE)
        axis.axvline(_nines(REF_TARGET), color=INK_2, lw=1, ls=(0, (1, 3)), zorder=1)
        axis.axvline(_nines(opt), color=INK_2, lw=1, ls=(0, (1, 3)), zorder=1)
        axis.grid(axis="y", color=GRID, lw=1)
        axis.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
    for m in MODELS:
        d = c[c["model"] == m].sort_values("target")
        lw = 2.4 if m in FOCUS else 1.4
        ax.plot(_nines(d["target"]), d["cost"], color=COLORS[m], lw=lw, label=m, zorder=3)
        ax2.plot(_nines(d["target"]), d["csl"], color=COLORS[m], lw=lw, zorder=3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Mean cost per 28-day fold", color=INK_2)
    ax.legend(
        loc="upper right",
        frameon=False,
        labelcolor=INK,
        fontsize=9.5,
        ncol=1,
        bbox_to_anchor=(0.86, 0.78),
    )
    top = ax.get_ylim()[1]
    ax.text(
        _nines(REF_TARGET) + 0.03,
        top * 0.985,
        "95%\n(the default)",
        va="top",
        fontsize=9,
        color=INK_2,
    )
    ax.text(
        _nines(opt) + 0.03,
        top * 0.985,
        f"cost-optimal\n{opt:.4%}",
        va="top",
        fontsize=9,
        color=INK_2,
    )
    if cr is not None:
        ax2.axhline(cr, color=INK, lw=1, zorder=2)
        ax2.text(
            _nines(LEVELS[0]) + 0.02,
            cr + 0.004,
            f"critical ratio {cr:.1%}",
            fontsize=9,
            color=INK,
            va="bottom",
        )
    ax2.set_ylim(0.65, 1.0)
    ax2.set_ylabel("Realised cycle\nservice level", color=INK_2)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ticks = [0.80, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
    ax2.xaxis.set_major_locator(FixedLocator(_nines(ticks)))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * (1 - 10 ** (-v)):.10g}%"))
    ax2.set_xlabel("Nominal service target (each tick adds a nine)", color=INK_2)
    fig.text(
        0.012,
        0.965,
        "The cost-optimal service target is far above 95%",
        color=INK,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.012,
        0.925,
        "Shipped safety-stock scheme, all five models, holding 25%/yr, "
        "lost sale = gross margin.",
        color=INK_2,
        fontsize=10,
        va="top",
    )
    fig.subplots_adjust(top=0.88, left=0.12, right=0.97, bottom=0.09)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---- report ----------------------------------------------------------------------------------


def _md(df: pd.DataFrame) -> str:
    return bt._markdown_table(df, float_cols=())


def _ci(r: pd.Series) -> str:
    span = f"[{r['ci_lo']:+,.0f}, {r['ci_hi']:+,.0f}]"
    return f"{r['point']:+,.0f} {span}" + (" — crosses zero" if r["crosses_zero"] else "")


def analyse_basis(detail: pd.DataFrame, annual: float, margin: float | None) -> dict:
    p = priced(detail, annual, margin)
    c = curves(p)
    opt = argmin_target(c)
    return {"p": p, "c": c, "opt": opt}


def build_report(
    detail: pd.DataFrame,
    costs: CostParams,
    prices: pd.Series,
    cycle_days: float,
    chart_rel: str,
    decomposition: list[dict] | None = None,
    saturation: pd.DataFrame | None = None,
) -> tuple[str, dict]:
    annual = DEFAULT_ANNUAL_HOLDING_RATE
    econ_b = analyse_basis(detail, annual, costs.gross_margin)
    flat_b = analyse_basis(detail, annual, None)
    ct = implied_targets(costs, prices, cycle_days, annual)

    cr_econ = float(
        ct[
            ct["pricing"].str.startswith("unit econ")
            & ct["protection period"].str.startswith("lead")
        ]["CR at mean price"].iloc[0]
    )
    cr_flat = float(
        ct[ct["pricing"].str.startswith("legacy") & ct["protection period"].str.startswith("lead")][
            "CR at mean price"
        ].iloc[0]
    )
    cr_flat_median = float(
        ct[ct["pricing"].str.startswith("legacy") & ct["protection period"].str.startswith("lead")][
            "CR median"
        ].iloc[0]
    )
    c = econ_b["c"]
    opt = econ_b["opt"]["target"]
    pooled_c = pooled(c)
    max_csl = float(pooled_c["csl"].max())

    curve_rows = pooled_c[
        pooled_c["target"].isin([0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
    ]
    curve_tbl = pd.DataFrame(
        {
            "nominal target": [f"{t:.4%}" for t in curve_rows["target"]],
            "z": [f"{norm.ppf(t):.2f}" for t in curve_rows["target"]],
            "mean cost / fold": [f"${v:,.0f}" for v in curve_rows["cost"]],
            "holding": [f"${v:,.0f}" for v in curve_rows["holding"]],
            "stockout": [f"${v:,.0f}" for v in curve_rows["stockout"]],
            "realised CSL": [f"{v:.1%}" for v in curve_rows["csl"]],
            "fill rate": [f"{v:.1%}" for v in curve_rows["fill"]],
        }
    )
    per_model = pd.DataFrame(
        [
            {
                "model": m,
                "cost-optimal nominal target": f"{argmin_target(c, m)['target']:.4%}"
                + (" (grid edge)" if argmin_target(c, m)["at_edge"] else ""),
                "cost at optimum": f"${argmin_target(c, m)['cost']:,.0f}",
                "realised CSL there": f"{argmin_target(c, m)['csl']:.1%}",
            }
            for m in MODELS
        ]
    )

    targets = [0.90, REF_TARGET, 0.99, 0.999, opt]
    targets = sorted(set(round(t, 8) for t in targets))
    rank = ranking_table(c, targets)
    cost_w = rank.pivot(index="model", columns="target", values="cost").loc[MODELS]
    rank_w = rank.pivot(index="model", columns="target", values="rank").loc[MODELS]
    hdr = {t: f"{t:.4%}" for t in targets}
    cost_w = cost_w.rename(columns=hdr).map("${:,.0f}".format).reset_index()
    rank_w = rank_w.rename(columns=hdr).reset_index()

    cmp95 = compare_at(econ_b["p"], REF_TARGET)
    cmpopt = compare_at(econ_b["p"], opt)
    cmp_tbl = pd.DataFrame(
        {
            "LightGBM minus": cmp95["other"],
            f"at {REF_TARGET:.0%}": [_ci(r) for _, r in cmp95.iterrows()],
            f"at {opt:.4%} (optimum)": [_ci(r) for _, r in cmpopt.iterrows()],
        }
    )
    sn95 = cmp95[cmp95["other"] == "SeasonalNaive"].iloc[0]
    snopt = cmpopt[cmpopt["other"] == "SeasonalNaive"].iloc[0]
    flips = np.sign(sn95["point"]) != np.sign(snopt["point"])

    flat_opt = flat_b["opt"]
    flat_cmp = compare_at(flat_b["p"], flat_opt["target"])
    flat_sn = flat_cmp[flat_cmp["other"] == "SeasonalNaive"].iloc[0]

    reach = nominal_for_csl(c, cr_econ)
    flat_at = flat_from(c)
    p_c = pooled(c)
    old_edge_gap = float(p_c[at_target(p_c, 0.999)]["cost"].iloc[0] / p_c["cost"].min() - 1)
    z_opt = float(norm.ppf(opt))
    interior = not econ_b["opt"]["at_edge"]
    lines = [
        f"# The cost-optimal service target, and the model ranking there — {dt.date.today()}",
        "",
        f"Holding {annual:.0%}/yr; lost sale = {costs.gross_margin:.1%} gross margin, stock valued "
        f"at cost (`docs/decision.md`). Shipped safety-stock scheme, all five models, 400 series x "
        f"4 folds, the nominal-target grid extended from 99.9% to 99.9999%. Costs are means per "
        f"28-day fold; CIs are paired bootstrap over series (2,000 resamples).",
        "",
        f"![Cost and realised service against the nominal target]({chart_rel})",
        "",
        "## 1. The implied optimum (critical ratio)",
        "",
        "`CR = Cu / (Cu + Co)`, `Cu` the per-unit lost-sale cost, `Co = rate x unit cost x "
        "period`. It is the cycle service level that minimises expected cost. Mean "
        f"training-window price ${prices.mean():.2f}; measured mean replenishment cycle "
        f"{cycle_days:.1f} days.",
        "",
        _md(ct.assign(**{k: ct[k].map("{:.2%}".format) for k in ct.columns if k.startswith("CR")})),
        "",
        f"**The implied optimal service level is ~{cr_econ:.1%} under unit economics** "
        f"(higher, ~{cr_flat:.1%} at mean price, under the legacy flat $5 — where the median SKU "
        f"sits at {cr_flat_median:.1%} — a ~99.6% figure describes that SKU, not the mean price). "
        "95% was not close to either. "
        "Because Co scales with price and so does Cu under unit economics, the ratio no longer "
        "varies across SKUs there (the p10/p90 columns collapse) — a uniform target is exactly "
        "right under that cost model, which the flat penalty had made look wrong.",
        "",
        "## 2. Where the sweep's optimum is, on the extended grid",
        "",
        _md(curve_tbl),
        "",
        f"Cost-minimising nominal target (pooled over the five models): **{opt:.4%}**, "
        + ("an interior optimum" if interior else "still at the grid edge")
        + f"; the curve is within 2% of that minimum from {flat_at:.4%} nominal, so the optimum "
        "is 'at least this high' rather than a sharp point. Realised CSL at the minimum is "
        f"{econ_b['opt']['csl']:.1%} against a critical ratio of {cr_econ:.1%}. "
        + (
            f"The smallest grid target whose realised CSL reaches it is {reach:.4%}."
            if reach is not None
            else f"**No grid target reaches the critical ratio: realised CSL tops out at "
            f"{max_csl:.1%}** even at 99.9999% nominal."
        ),
        "",
        "Per model:",
        "",
        _md(per_model),
        "",
        "## 3. Does the ranking survive at the optimum?",
        "",
        "Mean cost per fold and rank, by nominal target:",
        "",
        _md(cost_w),
        "",
        _md(rank_w),
        "",
        "LightGBM minus each other model (negative = LightGBM cheaper), paired bootstrap:",
        "",
        _md(cmp_tbl),
        "",
        (
            f"**The LightGBM/SeasonalNaive comparison flips sign between {REF_TARGET:.0%} "
            f"({sn95['point']:+,.0f}) and the optimum ({snopt['point']:+,.0f}).** Model choice "
            "depends on the service target as well as the holding rate."
            if flips
            else f"The LightGBM/SeasonalNaive comparison keeps its sign from {REF_TARGET:.0%} "
            f"({sn95['point']:+,.0f}) to the optimum ({snopt['point']:+,.0f})"
            + (
                ", and the interval at the optimum "
                + ("crosses zero." if snopt["crosses_zero"] else "excludes zero.")
            )
        ),
        "",
        f"Under the legacy flat $5 penalty the optimum is {flat_opt['target']:.4%}"
        + (" (grid edge)" if flat_opt["at_edge"] else "")
        + f", and LightGBM minus SeasonalNaive there is {_ci(flat_sn)}.",
        "",
    ]
    if decomposition:
        dec_tbl = pd.DataFrame(
            [
                {
                    "nominal target": f"{r['target']:.4%}",
                    "cycles": f"{r['cycles']:,}",
                    "stockout cycles": f"{r['stockout_cycles']:,}",
                    "demand exceeded s (sizing)": f"{r['sizing_failures']:,} "
                    f"({r['sizing_failures'] / r['stockout_cycles']:.0%})",
                    "demand <= s, on-hand < demand (undershoot)": f"{r['undershoot']:,} "
                    f"({r['undershoot'] / r['stockout_cycles']:.0%})",
                    "CSL if only sizing failed": f"{r['csl_if_sizing_only']:.2%}",
                }
                for r in decomposition
            ]
        )
        lines += [
            "## 4. Why realised service saturates below the critical ratio",
            "",
            "The earlier sweeps stopped at 99.9%, where pooled cost was still "
            f"{old_edge_gap:.0%} above its minimum: the edge they hit was the cost structure (a "
            "lost sale that dwarfs carrying a unit), not the policy. Realised service topping out "
            "*below* the critical "
            "ratio is a separate fact and needs its own explanation. Every stockout cycle across "
            "all models, folds and series is classified below (replicating the simulation's cycle "
            "logic; the replicated cycle and stockout counts equal the stored ones exactly):",
            "",
            _md(dec_tbl),
            "",
            "An order triggers when on-hand falls *below* s, so the stock held when it is placed "
            "is below s, and under `S = s` it replaces only that deficit with one order in "
            "transit at a time. Most stockout cycles are cycles where the buffer *would* have "
            "covered lead-time demand — a wider buffer buys little against them, which is why "
            "realised service saturates near 97% however high the nominal target goes. That the "
            "*formula* is not the constraint is checked directly below; that the *policy "
            "structure* is, is tested by changing it in `reorderpoint/order_up_to.py`. The "
            f"cost-optimal *nominal* target ({opt:.4%}, z = {z_opt:.2f}) is a symptom of paying "
            "for buffer to compensate, not a target anyone should ship.",
            "",
        ]
        if saturation is not None:
            top_t = saturation["target"].max()
            top_csl = saturation[saturation["target"] == top_t]["realised_csl"]
            sat = saturation.assign(
                scheme=saturation["form"] + " × " + saturation["granularity"],
                target=saturation["target"].map("{:.4%}".format),
                **{
                    "realised CSL": saturation["realised_csl"].map("{:.1%}".format),
                    "fill rate": saturation["fill"].map("{:.1%}".format),
                },
            )
            lines += [
                "Realised service by safety-stock scheme (`S = s`; `scheme_saturation`, "
                "reproducible with `make optimal-target`) — does a better sizing formula lift "
                "the ceiling?",
                "",
                _md(sat[["scheme", "target", "realised CSL", "fill rate"]]),
                "",
                f"Every scheme is capped: at {top_t:.4%} nominal the best realised CSL is "
                f"{top_csl.max():.1%} and the worst {top_csl.min():.1%}. "
                "(The empirical form cannot exceed what its largest calibration residual "
                "covers, so it plateaus by construction; finer pooling helps a little.)",
                "",
            ]
    stats = {
        "cr_econ": cr_econ,
        "opt": opt,
        "interior": interior,
        "max_csl": max_csl,
        "reach": reach,
        "flips": bool(flips),
        "sn95": sn95.to_dict(),
        "snopt": snopt.to_dict(),
        "flat_opt": flat_opt,
        "curves": c,
        "cr_table": ct,
    }
    return "\n".join(lines), stats


def main() -> None:
    costs = load_config().costs
    detail = run_sweep(costs)
    panel = bt.load_eval_panel()
    last_train = panel[panel["date"] <= bt.make_folds(panel)[-1].train_end]
    from reorderpoint import decision as dec

    prices = dec.unit_cost_from_price(last_train)
    base95 = detail[at_target(detail, REF_TARGET)]
    cycle_days = float(sw.realised_cycle_days(base95, bt.HORIZON))

    files = bt.REPORTS_DIR / "optimal_target_files"
    files.mkdir(parents=True, exist_ok=True)
    econ_curves = curves(priced(detail, DEFAULT_ANNUAL_HOLDING_RATE, costs.gross_margin))
    opt_target = argmin_target(econ_curves)["target"]
    decomposition = [
        stockout_decomposition(detail, panel, t, costs.lead_time_days)
        for t in dict.fromkeys((REF_TARGET, opt_target, LEVELS[-1]))
    ]
    forecasts = pd.read_parquet(dv.FORECASTS_PATH)
    calib = ab.calibration_residuals(panel, costs)
    saturation = scheme_saturation(costs, panel, forecasts, calib)
    report, stats = build_report(
        detail,
        costs,
        prices,
        cycle_days,
        "optimal_target_files/cost_vs_target.png",
        decomposition,
        saturation,
    )
    plot_target_curves(
        stats["curves"], stats["cr_econ"], stats["opt"], files / "cost_vs_target.png"
    )
    out = bt.REPORTS_DIR / f"optimal_target_{dt.date.today().isoformat()}.md"
    out.write_text(report)
    print(f"Wrote {out}")
    print({k: v for k, v in stats.items() if k not in ("curves", "cr_table", "sn95", "snopt")})


if __name__ == "__main__":
    main()
