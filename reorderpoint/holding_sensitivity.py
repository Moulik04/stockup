"""How much of the project's cost story depends on `holding_cost_rate`, whose unit was never stated?

`HOLDING_COST_RATE=0.02` entered in the initial commit as a config default ("illustrative, not real
business numbers"). Nothing in the repo or its history says what period 2% refers to. The one
unit-bearing statement is the simulation itself, which accrues it **per day** on end-of-day
on-hand (`decision.simulate_series`, documented in `docs/decision.md`) — i.e. ~730% of unit cost
per year. That is not a plausible carrying cost, so the question is what happens to the results
under readings that are:

- **2%/day** — the legacy default, the value every Phase 4 number was produced at.
- **2%/month** (~24%/yr) — inside the textbook 15–30%/yr carrying-cost range.
- **2%/year** — the literal annual reading; below that range, so a lower bound, not a best guess.
- **25%/yr** — the default adopted 2026-09-20 (`docs/decision.md`, "Holding cost rate").

`holding_breakeven.py` goes further: it solves for the rate at which the LightGBM/SeasonalNaive
ranking flips (~364%/yr) instead of sampling it at these points.

**Why this can be done without resimulating.** The safety-stock policy never reads the holding
rate (`docs/decision.md`: it "only prices the simulation"), and the simulation accrues holding as
`rate × unit_cost × on_hand`, linear in the rate. Every stored run therefore carries both cost
components, and re-pricing is exact: `total(h) = holding × h/h₀ + stockout`. `reprice` asserts that
identity reproduces the stored total at the rate the run used before anything is trusted.

What it re-answers, for every rate: the model cost ranking (with the LightGBM-vs-baselines CIs),
the calibration-scheme comparison, and the cost-optimal service level.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import decision as dec
from reorderpoint import service_level_sweep as sw
from reorderpoint.config import (
    DEFAULT_ANNUAL_HOLDING_RATE,
    CostParams,
    annual_to_daily,
    describe_holding_rate,
    load_config,
)
from reorderpoint.holding_breakeven import LEGACY_SIMULATED_RATE, simulated_rate

CONFIGURED = "2%/day (legacy default)"
MONTHLY = "2%/month (~24%/yr)"
ANNUAL = "2%/year"
DEFAULT = f"{DEFAULT_ANNUAL_HOLDING_RATE:.0%}/yr (default)"
RATES = {
    CONFIGURED: LEGACY_SIMULATED_RATE,
    MONTHLY: 0.02 / 30,
    ANNUAL: 0.02 / 365,
    DEFAULT: annual_to_daily(DEFAULT_ANNUAL_HOLDING_RATE),
}
MODELS = ["LightGBM", "AutoETS", "AutoTheta", "MovingAverage", "SeasonalNaive"]

# Phase 7's NBEATS ran on Bridges-2 and only aggregate tables came back (README, decision_deep
# report), so it can be re-priced as a point estimate but not bootstrapped.
NBEATS_AGG = {"holding": 5960.687, "stockout": 11248.678}


def reprice(df: pd.DataFrame, rate: float, base_rate: float) -> pd.DataFrame:
    """`total_cost` re-priced at `rate`, given the run was simulated at `base_rate`."""
    return df.assign(total_cost=df["holding_cost"] * (rate / base_rate) + df["stockout_cost"])


def check_reprice_identity(df: pd.DataFrame, base_rate: float) -> None:
    """Re-pricing at the rate the run used must give back the stored total, or the
    linear-in-rate premise (and so everything below) is wrong."""
    rebuilt = reprice(df, base_rate, base_rate)["total_cost"]
    if not np.allclose(rebuilt, df["total_cost"], rtol=1e-9, atol=1e-6):
        raise ValueError("holding_cost + stockout_cost does not reproduce total_cost")


def model_costs(detail: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    """Mean cost per fold per model at every rate (CSL policy), plus the ranking at each."""
    d = detail[detail["policy"] == "csl"]
    check_reprice_identity(d, base_rate)
    rows = []
    for label, rate in RATES.items():
        r = reprice(d, rate, base_rate)
        per = r.groupby(["model", "fold"])["total_cost"].sum().groupby("model").mean()
        rows += [{"rate": label, "model": m, "cost": c} for m, c in per.items()]
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("rate")["cost"].rank().astype(int)
    return out


def model_comparisons(detail: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    """LightGBM minus each other model, per fold, with the paired-bootstrap CI, at every rate."""
    d = detail[detail["policy"] == "csl"]
    rows = []
    for label, rate in RATES.items():
        r = reprice(d, rate, base_rate)
        for other in MODELS[1:]:
            out = boot.bootstrap_diff(
                r[r["model"] == "LightGBM"],
                r[r["model"] == other],
                "total_cost",
                group_cols=["fold"],
            )
            rows.append({"rate": label, "other": other, **out})
    return pd.DataFrame(rows)


def scheme_comparisons(cell_detail: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    """Each finer scheme vs the shipped one at the same 95% nominal target, at every rate, split
    into its holding and stockout components. If both components fall, the scheme wins at *any*
    holding rate — no re-pricing can undo it."""
    base = cell_detail[
        (cell_detail["form"] == ab.BASELINE_CELL[0])
        & (cell_detail["granularity"] == ab.BASELINE_CELL[1])
    ]
    rows = []
    for cell in [("normal", "volume_quintile"), ("normal", "volume_tercile_within_intermittency")]:
        alt = cell_detail[
            (cell_detail["form"] == cell[0]) & (cell_detail["granularity"] == cell[1])
        ]
        check_reprice_identity(alt, base_rate)
        for label, rate in RATES.items():
            out = boot.bootstrap_diff(
                reprice(alt, rate, base_rate),
                reprice(base, rate, base_rate),
                "total_cost",
                group_cols=["fold", "model"],
            )
            rows.append({"rate": label, "cell": cell[1], **out})
    return pd.DataFrame(rows)


def scheme_components(cell_detail: pd.DataFrame) -> pd.DataFrame:
    """Per-fold-per-model mean holding and stockout cost of each cell, at its simulated rate."""
    g = cell_detail.groupby(["form", "granularity", "model", "fold"])[
        ["holding_cost", "stockout_cost"]
    ].sum()
    return g.groupby(["form", "granularity"]).mean().reset_index()


def optimal_targets(
    fold_rows: pd.DataFrame, base_rate: float, folds: list[int] | None = None
) -> pd.DataFrame:
    """Cost-minimising nominal target per (scheme, rate), from the sweep's per-fold rows.

    `folds` restricts which folds the choice is made on — the hook the hold-out uses.
    """
    rows = fold_rows if folds is None else fold_rows[fold_rows["fold"].isin(folds)]
    out = []
    for label, rate in RATES.items():
        r = reprice(rows, rate, base_rate)
        curve = r.groupby(["form", "granularity", "target"]).agg(
            cost=("total_cost", "mean"), csl=("cycle_service_level", "mean")
        )
        for (form, gran), g in curve.reset_index().groupby(["form", "granularity"]):
            best = g.loc[g["cost"].idxmin()]
            at95 = g[np.isclose(g["target"], 0.95)].iloc[0]
            out.append(
                {
                    "rate": label,
                    "form": form,
                    "granularity": gran,
                    "argmin_target": float(best["target"]),
                    "argmin_cost": float(best["cost"]),
                    "realised_csl_at_argmin": float(best["csl"]),
                    "saving_vs_95": float((at95["cost"] - best["cost"]) / at95["cost"]),
                    "at_grid_edge": bool(
                        best["target"] >= g["target"].max() - 1e-9
                        or best["target"] <= g["target"].min() + 1e-9
                    ),
                }
            )
    return pd.DataFrame(out)


def critical_ratios(costs: CostParams, unit_costs: pd.Series, cycle_days: float) -> pd.DataFrame:
    rows = []
    for label, rate in RATES.items():
        c = replace(costs, holding_cost_rate=rate)
        per = sw.critical_ratio(c, unit_costs.to_numpy(), cycle_days)
        rows.append(
            {
                "rate": label,
                "cr_mean_price": float(sw.critical_ratio(c, unit_costs.mean(), cycle_days)),
                "cr_p10": float(np.quantile(per, 0.1)),
                "cr_median": float(np.median(per)),
                "cr_p90": float(np.quantile(per, 0.9)),
            }
        )
    return pd.DataFrame(rows)


def nbeats_repriced() -> pd.DataFrame:
    rows = []
    for label, rate in RATES.items():
        k = rate / LEGACY_SIMULATED_RATE
        rows.append({"rate": label, "NBEATS": NBEATS_AGG["holding"] * k + NBEATS_AGG["stockout"]})
    return pd.DataFrame(rows)


def ranking_is_stable(ranks: pd.DataFrame) -> bool:
    orders = {label: tuple(g.sort_values("rank")["model"]) for label, g in ranks.groupby("rate")}
    return len(set(orders.values())) == 1


PROVENANCE = """## 1. Where the value came from, and what it was meant to be

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
"""


def build_report(
    ranks: pd.DataFrame,
    comps: pd.DataFrame,
    schemes: pd.DataFrame,
    components: pd.DataFrame,
    optima: pd.DataFrame | None,
    crs: pd.DataFrame,
    nbeats: pd.DataFrame,
    costs: CostParams,
    simulated_label: str = describe_holding_rate(LEGACY_SIMULATED_RATE),
) -> tuple[str, dict]:
    stable = ranking_is_stable(ranks)
    wide = ranks.pivot(index="model", columns="rate", values="cost")[list(RATES)]
    rank_wide = ranks.pivot(index="model", columns="rate", values="rank")[list(RATES)]
    winners = {label: g.sort_values("rank").iloc[0]["model"] for label, g in ranks.groupby("rate")}

    def ci_txt(r: pd.Series) -> str:
        span = f"[{r['ci_lo']:+,.0f}, {r['ci_hi']:+,.0f}]"
        return f"{r['point']:+,.0f} {span}" + (" — crosses zero" if r["crosses_zero"] else "")

    comp_view = comps.assign(
        diff_ci=[ci_txt(r) for _, r in comps.iterrows()],
    )[
        ["rate", "other", "diff_ci"]
    ].pivot(index="other", columns="rate", values="diff_ci")[list(RATES)]

    lgb_vs_sn = comps[comps["other"] == "SeasonalNaive"].set_index("rate")
    lgb_wins_everywhere = all(
        (not r["crosses_zero"]) and r["point"] < 0 for _, r in lgb_vs_sn.iterrows()
    )

    lines = [
        f"# Holding-cost sensitivity — {dt.date.today().isoformat()}",
        "",
        "Every cost in the project prices holding at `holding_cost_rate` per unit-cost per "
        "**day**. That rate's intended unit was never recorded, and at the legacy default it is "
        "not a plausible carrying cost. This report re-prices the existing runs at three readings "
        "of '2%' and at the 25%/yr default, and asks which conclusions move. Method and why no "
        "resimulation is needed: see the module "
        "docstring of `reorderpoint/holding_sensitivity.py`.",
        "",
        PROVENANCE,
        "## 2. Does the model cost ranking move?",
        "",
        "Mean simulated cost per fold, CSL policy, shipped scheme, same 400 series / 4 folds:",
        "",
        bt._markdown_table(
            wide.reset_index().rename(columns={"model": "model"}),
            float_cols=tuple(RATES),
        ),
        "",
        "Rank (1 = cheapest):",
        "",
        bt._markdown_table(rank_wide.reset_index(), float_cols=()),
        "",
        "LightGBM minus each other model, per fold, paired-bootstrap 95% CI over series "
        "(negative = LightGBM cheaper):",
        "",
        bt._markdown_table(comp_view.reset_index(), float_cols=()),
        "",
    ]

    nb = nbeats.set_index("rate")["NBEATS"]
    lines += [
        "NBEATS (point estimates only — no per-series data to bootstrap, see README): "
        + ", ".join(f"{label} ${nb[label]:,.0f}" for label in RATES)
        + "; SeasonalNaive at the same rates: "
        + ", ".join(f"${wide.loc['SeasonalNaive', label]:,.0f}" for label in RATES)
        + ".",
        "",
    ]

    if stable:
        lines += [
            "**The ranking is stable across all three readings** — same order at every rate.",
            "",
        ]
    else:
        lines += [
            "**The ranking is not stable.** "
            + "; ".join(f"cheapest at {label}: **{w}**" for label, w in winners.items())
            + ". "
            + (
                "LightGBM still beats SeasonalNaive at every rate."
                if lgb_wins_everywhere
                else "LightGBM's cost win over SeasonalNaive does not hold at every rate — see "
                "the CI column: at the lower holding rates stockouts dominate the total, and "
                "SeasonalNaive is the model that stocks out least (its oscillating forecast is an "
                "unpriced safety margin, README Phase 7)."
            ),
            "",
        ]

    lines += [
        "## 3. Does the calibration-scheme win move?",
        "",
        "Finer scheme minus shipped scheme, both at the same 95% nominal target, mean per fold "
        "(negative = cheaper), with paired-bootstrap CI:",
        "",
        bt._markdown_table(
            schemes.assign(diff_ci=[ci_txt(r) for _, r in schemes.iterrows()])
            .pivot(index="cell", columns="rate", values="diff_ci")[list(RATES)]
            .reset_index(),
            float_cols=(),
        ),
        "",
        f"The components at the rate they were simulated at ({simulated_label}), per fold "
        "(holding / stockout):",
        "",
        bt._markdown_table(
            components.assign(
                holding_cost=components["holding_cost"].round(0),
                stockout_cost=components["stockout_cost"].round(0),
            ),
            float_cols=(),
        ),
        "",
    ]
    base_c = components[
        (components["form"] == ab.BASELINE_CELL[0])
        & (components["granularity"] == ab.BASELINE_CELL[1])
    ].iloc[0]
    q_c = components[
        (components["form"] == "normal") & (components["granularity"] == "volume_quintile")
    ].iloc[0]
    both_lower = q_c["holding_cost"] < base_c["holding_cost"] and (
        q_c["stockout_cost"] < base_c["stockout_cost"]
    )
    q_rows = schemes[schemes["cell"] == "volume_quintile"]
    sign_robust = bool(both_lower and (q_rows["point"] < 0).all())
    sig_everywhere = bool((~q_rows["crosses_zero"]).all())
    scheme_robust = sign_robust and sig_everywhere
    t_rows = schemes[schemes["cell"] == "volume_tercile_within_intermittency"]
    tercile_sig = bool((~t_rows["crosses_zero"]).all())
    size = {r["rate"]: r["point"] for _, r in q_rows.iterrows()}
    if scheme_robust:
        lines += [
            "**The scheme win is robust to the holding rate.** The volume-quintile scheme has "
            "lower "
            f"holding (${q_c['holding_cost']:,.0f} vs ${base_c['holding_cost']:,.0f}) *and* lower "
            f"stockout cost (${q_c['stockout_cost']:,.0f} vs ${base_c['stockout_cost']:,.0f}) at "
            "the same target, so no re-weighting of the two can reverse its sign — and its CI "
            "excludes zero at every rate.",
            "",
        ]
    elif sign_robust:
        lines += [
            "**The scheme win keeps its sign at every rate but not its significance or its size.** "
            f"The volume-quintile scheme has lower holding (${q_c['holding_cost']:,.0f} vs "
            f"${base_c['holding_cost']:,.0f}) *and* lower stockout cost "
            f"(${q_c['stockout_cost']:,.0f} vs ${base_c['stockout_cost']:,.0f}) at the same "
            "target, so re-weighting the two cannot reverse the direction. But the saving shrinks "
            f"from ${abs(size[CONFIGURED]):,.0f} to ${abs(size[MONTHLY]):,.0f} per fold at "
            f"2%/month and ${abs(size[ANNUAL]):,.0f} at 2%/year, and the interval at the lower "
            "rates is at or across zero: the 11% headline saving was mostly a holding-cost "
            "saving, and holding is the term that is mispriced. "
            + (
                "The tercile scheme, by contrast, stays significant at every rate."
                if tercile_sig
                else ""
            ),
            "",
        ]
    else:
        lines += [
            "**The scheme win is not robust to the holding rate** — see the intervals above.",
            "",
        ]

    if optima is not None:
        view = optima.assign(
            scheme=optima["form"] + " × " + optima["granularity"],
            argmin=[
                f"{a:.1%}" + (" (grid edge)" if e else "")
                for a, e in zip(optima["argmin_target"], optima["at_grid_edge"], strict=True)
            ],
            realised_csl=optima["realised_csl_at_argmin"].map(lambda v: f"{v:.1%}"),
            saving_vs_95=optima["saving_vs_95"].map(lambda v: f"{v:.1%}"),
        )[["rate", "scheme", "argmin", "realised_csl", "saving_vs_95"]]
        lines += [
            "## 4. Does the cost-optimal service level move?",
            "",
            "Nominal-target argmin of simulated cost, per scheme and rate (grid 80%–99.9%):",
            "",
            bt._markdown_table(view, float_cols=()),
            "",
            "Newsvendor critical ratio at each rate (the optimal *realised* CSL; mean price with "
            "the measured cycle length, and its spread across SKUs):",
            "",
            bt._markdown_table(crs, float_cols=("cr_mean_price", "cr_p10", "cr_median", "cr_p90")),
            "",
        ]
        base_opt = optima[
            (optima["form"] == ab.BASELINE_CELL[0]) & (optima["granularity"] == ab.BASELINE_CELL[1])
        ].set_index("rate")
        targets = base_opt["argmin_target"]
        lines += [
            f"Under the shipped scheme the cost-minimising target is {targets[CONFIGURED]:.1%} at "
            f"2%/day, {targets[MONTHLY]:.1%} at 2%/month, {targets[ANNUAL]:.1%} at 2%/year and "
            f"{targets[DEFAULT]:.1%} at the 25%/yr default. "
            + (
                "**It moves substantially with the holding rate**, so 'the optimum is ~92%' "
                "(Task 7) was a statement about the legacy 2%/day rate, not about the problem."
                if targets.max() - targets.min() > 0.03
                else "It barely moves."
            ),
            "",
        ]

    return "\n".join(lines), {
        "stable": stable,
        "winners": winners,
        "lgb_wins_everywhere": lgb_wins_everywhere,
        "scheme_robust": scheme_robust,
        "scheme_sign_robust": sign_robust,
    }


def main() -> None:
    # These analyses re-price stored runs that were simulated under the legacy cost model (flat
    # $5 penalty, stock at price), varying the holding rate only — so they stay on it. The unit-
    # economics treatment of the penalty is `penalty_sensitivity.py`.
    costs = load_config().costs.flat()
    # Each stored artifact is re-priced from the rate *it* was simulated at, never from the config
    # default — which changed on 2026-09-20 without the parquets changing.
    detail = pd.read_parquet(dec.DETAIL_PATH)
    cell_detail = pd.read_parquet(ab.CELL_DETAIL_PATH)
    detail_rate, cell_rate = simulated_rate(detail), simulated_rate(cell_detail)

    ranks = model_costs(detail, detail_rate)
    comps = model_comparisons(detail, detail_rate)
    schemes = scheme_comparisons(cell_detail, cell_rate)
    components = scheme_components(cell_detail)

    optima = crs = None
    if sw.SWEEP_ROWS_PATH.exists():
        fold_rows = pd.read_parquet(sw.SWEEP_ROWS_PATH)
        sweep_rate = simulated_rate(fold_rows)
        check_reprice_identity(fold_rows, sweep_rate)
        optima = optimal_targets(fold_rows, sweep_rate)
        panel = bt.load_eval_panel()
        last_train = panel[panel["date"] <= bt.make_folds(panel)[-1].train_end]
        unit_costs = dec.unit_cost_from_price(last_train)
        cycle_days = float(
            sw.realised_cycle_days(
                cell_detail[
                    (cell_detail["form"] == "normal")
                    & (cell_detail["granularity"] == "intermittency")
                ],
                bt.HORIZON,
            )
        )
        crs = critical_ratios(costs, unit_costs, cycle_days)
    else:
        print("no sweep rows on disk — skipping the service-level section")

    report, stats = build_report(
        ranks,
        comps,
        schemes,
        components,
        optima,
        crs if crs is not None else pd.DataFrame(),
        nbeats_repriced(),
        costs,
        describe_holding_rate(cell_rate),
    )
    out = bt.REPORTS_DIR / f"holding_cost_sensitivity_{dt.date.today().isoformat()}.md"
    out.write_text(report)
    print(f"Wrote {out}")
    print(stats)


if __name__ == "__main__":
    main()
