"""Is the ~97% target a real finding, or a fold-specific one?

The service-level sweep picked its target on the same four folds the result is then reported
against — the selection-on-the-test-set problem the bootstrap work exists to prevent. This holds
the **final fold** out: the target is chosen from folds 1–3 only, then scored on fold 4, a window
the choice never saw. (The safety-stock calibration already chains forward without leakage — each
fold is sized from the previous fold's residuals — so the only thing selected on the data is the
target itself.)

Two selection rules, both applied to folds 1–3 only:

- **Matched service:** the smallest grid target at which the finer scheme's mean realised CSL is at
  least the shipped scheme's at 95%. This is how 97% was picked, and it does not use cost.
- **Cost argmin:** the cost-minimising nominal target per scheme, which *does* depend on the
  holding rate — so it is reported at each reading (see `holding_sensitivity.py`).

**Pass criteria, fixed before the held-out numbers were looked at.** Adopting the finer scheme at
its train-selected target requires, on fold 4, against the shipped scheme at 95%:

1. cost: the paired-bootstrap 95% CI on the difference excludes zero, on the cheaper side;
2. CSL: the CI does not lie entirely below zero (no distinguishable service loss);
3. stability: the held-out optimum is within 2 points of nominal target of the train-selected one,
   *or* using the train-selected target instead costs < 1% more than the held-out optimum.

If the held-out optimum differs materially the curve is reported and 95% is kept, rather than
shipping a fold-specific value. The bootstrap here is over series within a single 28-day window, so
it says nothing about window-to-window variation; the walk-forward table (select on folds < k,
score fold k, for k = 2, 3, 4) is reported for that, as point estimates.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import divergence as dv
from reorderpoint import holding_sensitivity as hs
from reorderpoint import service_level_sweep as sw
from reorderpoint.config import CostParams, load_config
from reorderpoint.holding_breakeven import simulated_rate

FINE = ("normal", "volume_quintile")
BASE = ab.BASELINE_CELL
NOMINAL = 0.95
STABLE_POINTS = 0.02
STABLE_REGRET = 0.01


def _cell(rows: pd.DataFrame, cell: tuple[str, str]) -> pd.DataFrame:
    return rows[(rows["form"] == cell[0]) & (rows["granularity"] == cell[1])]


def matched_target(rows: pd.DataFrame, folds: list[int]) -> float | None:
    """Smallest grid target where FINE's mean CSL over `folds` >= BASE's at NOMINAL."""
    r = rows[rows["fold"].isin(folds)]
    base = _cell(r, BASE)
    base_csl = base[np.isclose(base["target"], NOMINAL)]["cycle_service_level"].mean()
    fine = _cell(r, FINE).groupby("target")["cycle_service_level"].mean()
    ok = fine[fine >= base_csl]
    return float(ok.index.min()) if len(ok) else None


def fold_curve(rows: pd.DataFrame, fold: int, rate: float, base_rate: float) -> pd.DataFrame:
    """Repriced mean cost/CSL/fill per (scheme, target) on one fold."""
    r = hs.reprice(rows[rows["fold"] == fold], rate, base_rate)
    return (
        r.groupby(["form", "granularity", "target"])
        .agg(
            cost=("total_cost", "mean"),
            csl=("cycle_service_level", "mean"),
            fill=("fill_rate", "mean"),
        )
        .reset_index()
    )


def walk_forward(rows: pd.DataFrame, base_rate: float, rate: float) -> pd.DataFrame:
    """Select the matched target on folds < k, score fold k against the shipped scheme at 95%.
    Point estimates: one 28-day window per row."""
    out = []
    for k in sorted(rows["fold"].unique())[1:]:
        tr = [f for f in sorted(rows["fold"].unique()) if f < k]
        target = matched_target(rows, tr)
        if target is None:
            continue
        c = fold_curve(rows, k, rate, base_rate)
        base = _cell(c, BASE)
        base = base[np.isclose(base["target"], NOMINAL)].iloc[0]
        fine = _cell(c, FINE)
        fine = fine[np.isclose(fine["target"], target)].iloc[0]
        out.append(
            {
                "scored_fold": int(k),
                "selected_on": f"folds {tr[0]}–{tr[-1]}" if len(tr) > 1 else f"fold {tr[0]}",
                "selected_target": target,
                "cost_diff": fine["cost"] - base["cost"],
                "csl_diff_pp": (fine["csl"] - base["csl"]) * 100,
                "fill_diff_pp": (fine["fill"] - base["fill"]) * 100,
            }
        )
    return pd.DataFrame(out)


def evaluate_heldout(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    calib: pd.DataFrame,
    costs: CostParams,
    fold: int,
    fine_target: float,
    rates: dict[str, float],
) -> pd.DataFrame:
    """Series-level paired bootstrap on the held-out fold: FINE at `fine_target` minus BASE at 95%,
    for cost (repriced at each rate), CSL and fill rate."""
    _, base_d, _ = ab.run_cell(
        panel, forecasts, calib, *BASE, costs, n_boot=20, service_level=NOMINAL
    )
    _, fine_d, _ = ab.run_cell(
        panel, forecasts, calib, *FINE, costs, n_boot=20, service_level=fine_target
    )
    base_d = base_d[base_d["fold"] == fold].assign(
        cycles_ok=lambda d: d["cycles"] - d["stockout_cycles"]
    )
    fine_d = fine_d[fine_d["fold"] == fold].assign(
        cycles_ok=lambda d: d["cycles"] - d["stockout_cycles"]
    )
    g = ["fold", "model"]
    rows = []
    for label, rate in rates.items():
        out = boot.bootstrap_diff(
            hs.reprice(fine_d, rate, costs.holding_cost_rate),
            hs.reprice(base_d, rate, costs.holding_cost_rate),
            "total_cost",
            group_cols=g,
        )
        rows.append({"rate": label, "metric": "cost", **out})
    rows.append(
        {
            "rate": "—",
            "metric": "csl",
            **boot.bootstrap_ratio_diff(fine_d, base_d, "cycles_ok", "cycles", group_cols=g),
        }
    )
    rows.append(
        {
            "rate": "—",
            "metric": "fill_rate",
            **boot.bootstrap_ratio_diff(
                fine_d, base_d, "units_shipped", "units_demanded", group_cols=g
            ),
        }
    )
    return pd.DataFrame(rows)


def stability(train_opt: float, held_curve: pd.DataFrame, cell: tuple[str, str]) -> dict:
    """Held-out optimum vs the train-selected target, and the regret of using the latter."""
    c = _cell(held_curve, cell)
    best = c.loc[c["cost"].idxmin()]
    at_sel = c.iloc[(c["target"] - train_opt).abs().argmin()]
    regret = (at_sel["cost"] - best["cost"]) / best["cost"]
    stable = abs(best["target"] - train_opt) <= STABLE_POINTS + 1e-9 or regret < STABLE_REGRET
    return {"held_opt": float(best["target"]), "regret": float(regret), "stable": bool(stable)}


def build_report(
    rows: pd.DataFrame,
    base_rate: float,
    held: pd.DataFrame,
    heldout_fold: int,
    train_folds: list[int],
    target: float | None,
    train_optima: pd.DataFrame,
    stab: dict[str, dict],
) -> tuple[str, dict]:
    def ci(r: pd.Series, scale: float = 1.0, prefix: str = "", suffix: str = "") -> str:
        f = lambda v: f"{prefix}{v * scale:+,.1f}{suffix}"  # noqa: E731
        return f"{f(r['point'])} [{f(r['ci_lo'])}, {f(r['ci_hi'])}]" + (
            " — crosses zero" if r["crosses_zero"] else ""
        )

    csl = held[held["metric"] == "csl"].iloc[0]
    fill = held[held["metric"] == "fill_rate"].iloc[0]
    cost_rows = held[held["metric"] == "cost"].set_index("rate")

    csl_ok = not (csl["ci_hi"] < 0)
    verdicts = {}
    lines = [
        f"# Held-out validation of the service-level target — {dt.date.today().isoformat()}",
        "",
        f"Target chosen on folds {train_folds[0]}–{train_folds[-1]} only, scored on fold "
        f"{heldout_fold} (the most recent 28-day window; never seen by the selection). See the "
        "module docstring of `reorderpoint/holdout_target.py` for the pass criteria, which were "
        "fixed before the held-out numbers were read.",
        "",
        "## 1. Selection on the training folds",
        "",
        f"**Matched-service rule** (smallest target at which `{FINE[0]} × {FINE[1]}` realises at "
        f"least the shipped scheme's CSL at 95%, on folds {train_folds[0]}–{train_folds[-1]}): "
        + (f"**{target:.1%}**." if target is not None else "no grid target reaches it."),
        "",
        "**Cost-argmin rule**, per holding-rate reading:",
        "",
        bt._markdown_table(
            train_optima.assign(
                scheme=train_optima["form"] + " × " + train_optima["granularity"],
                argmin_target=train_optima["argmin_target"].map(lambda v: f"{v:.1%}"),
            )[["rate", "scheme", "argmin_target"]],
            float_cols=(),
        ),
        "",
    ]
    if target is None:
        return "\n".join(lines), {"adopt": False, "reason": "no matched target"}

    lines += [
        f"## 2. Held-out fold {heldout_fold}: `{FINE[0]} × {FINE[1]}` at {target:.1%} vs. the "
        f"shipped scheme at 95%",
        "",
        "Paired bootstrap over the 400 series within this one window (2,000 resamples), fine minus "
        "shipped, per model-fold row averaged over the five models:",
        "",
        "| metric | difference (95% CI) |",
        "|---|---|",
    ]
    for label in cost_rows.index:
        lines.append(
            f"| cost per fold, holding at {label} | {ci(cost_rows.loc[label], prefix='$')} |"
        )
    lines += [
        f"| cycle service level | {ci(csl, scale=100, suffix='pp')} |",
        f"| fill rate | {ci(fill, scale=100, suffix='pp')} |",
        "",
    ]

    for label, r in cost_rows.iterrows():
        verdicts[label] = bool((not r["crosses_zero"]) and r["point"] < 0)

    stab_rows = [
        {
            "rate": k,
            "train_selected": f"{v['train_opt']:.1%}",
            "held_out_optimum": f"{v['held_opt']:.1%}",
            "regret_of_selected": f"{v['regret']:.2%}",
            "stable": "yes" if v["stable"] else "**no**",
        }
        for k, v in stab.items()
    ]
    lines += [
        "## 3. Is the selected target stable? (`"
        + f"{FINE[0]} × {FINE[1]}"
        + "`, cost-argmin rule)",
        "",
        bt._markdown_table(pd.DataFrame(stab_rows), float_cols=()),
        "",
        "`regret_of_selected` is how much more the held-out fold costs at the train-selected "
        f"target than at the held-out optimum. Stable = within {STABLE_POINTS * 100:.0f} points "
        f"of nominal target, or regret < {STABLE_REGRET:.0%}.",
        "",
    ]
    return "\n".join(lines), {"verdicts": verdicts, "csl_ok": csl_ok, "csl": csl, "fill": fill}


def verdict_section(stats: dict, stab: dict[str, dict]) -> list[str]:
    """Each pre-registered criterion, per holding-rate reading, and what follows from them."""
    if "verdicts" not in stats:
        return [
            "## 6. Verdict",
            "",
            "No matched target on the training folds; nothing to adopt.",
            "",
        ]
    rows, adopt = [], []
    for label, cost_ok in stats["verdicts"].items():
        stable = stab[label]["stable"]
        ok = cost_ok and stats["csl_ok"] and stable
        rows.append(
            {
                "holding rate": label,
                "1. held-out cost CI excludes zero (cheaper)": "pass" if cost_ok else "**fail**",
                "2. no distinguishable CSL loss": "pass" if stats["csl_ok"] else "**fail**",
                "3. selected target stable": (
                    "pass — **vacuous: optimum at the grid edge**"
                    if stable and stab[label]["at_edge"]
                    else "pass" if stable else "**fail**"
                ),
                "adopt": "yes" if ok else "**no**",
            }
        )
        if ok:
            adopt.append(label)
    lines = [
        "## 6. Verdict against the pre-registered criteria",
        "",
        bt._markdown_table(pd.DataFrame(rows), float_cols=()),
        "",
    ]
    if len(adopt) == len(stats["verdicts"]):
        lines += ["All three criteria pass at every holding-rate reading."]
    elif adopt:
        lines += [
            f"The criteria pass only at: {', '.join(adopt)}. Because the holding rate's "
            "intended unit is unresolved, a finding that holds at some readings and not "
            "others is not enough to change the default."
        ]
    else:
        lines += [
            "The criteria do not pass at any reading. Keep the shipped default and report "
            "the curve."
        ]
    edge = [k for k, v in stab.items() if v["at_edge"]]
    if edge:
        lines += [
            "",
            f"**Read criterion 3 with care at {', '.join(edge)}.** There the cost-minimising "
            "target on the training folds and on the held-out fold are both the top of the grid "
            '(99.9%), so "stable" only means both clipped the same way — the analysis cannot '
            "say where the optimum is, only that it is higher than anything tried. What *is* "
            "established at every reading is criteria 1 and 2 for the specific 97% target, and "
            "the decomposition above shows how much of that saving is the scheme and how much "
            "is simply carrying more stock.",
        ]
    lines += [""]
    return lines


def decompose(
    rows: pd.DataFrame, folds: list[int], target: float, base_rate: float
) -> pd.DataFrame:
    """Split "finer scheme at `target` vs shipped at 95%" into its two changes: the scheme, and the
    higher target. At low holding rates carrying more stock is cheap, so a saving that looks like
    a scheme win can be mostly a target win — this says how much is which.

    Mean cost per fold over `folds` at each holding rate (point estimates; the CIs are in the
    sensitivity report)."""
    out = []
    for label, rate in hs.RATES.items():
        r = hs.reprice(rows[rows["fold"].isin(folds)], rate, base_rate)

        def cost(cell: tuple[str, str], t: float, r: pd.DataFrame = r) -> float:
            c = _cell(r, cell)
            return float(c[np.isclose(c["target"], t)]["total_cost"].mean())

        b95, bt_, f95, ft = (
            cost(BASE, NOMINAL),
            cost(BASE, target),
            cost(FINE, NOMINAL),
            cost(FINE, target),
        )
        out.append(
            {
                "holding rate": label,
                "shipped @95%": b95,
                f"shipped @{target:.0%}": bt_,
                "finer @95%": f95,
                f"finer @{target:.0%}": ft,
                "target effect (shipped)": bt_ - b95,
                "scheme effect (@ same target)": ft - bt_,
                "total": ft - b95,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    # The stored sweep rows were priced under the legacy cost model (flat penalty, stock at price);
    # the held-out re-simulation must use the same or its costs are not comparable with them.
    costs = load_config().costs.flat()
    rows = pd.read_parquet(sw.SWEEP_ROWS_PATH)
    # The sweep rows carry the rate they were simulated at; the config default is not it.
    base_rate = simulated_rate(rows)
    hs.check_reprice_identity(rows, base_rate)
    folds = sorted(rows["fold"].unique())
    heldout, train = int(folds[-1]), [int(f) for f in folds[:-1]]

    target = matched_target(rows, train)
    train_opt = hs.optimal_targets(rows, base_rate, folds=train)
    train_opt = train_opt[(train_opt["form"] == FINE[0]) & (train_opt["granularity"] == FINE[1])]

    panel = bt.load_eval_panel()
    forecasts = pd.read_parquet(dv.FORECASTS_PATH)
    calib = ab.calibration_residuals(panel, costs)

    held = None
    if target is not None:
        held = evaluate_heldout(panel, forecasts, calib, costs, heldout, target, hs.RATES)

    stab = {}
    for label, rate in hs.RATES.items():
        sel = float(train_opt[train_opt["rate"] == label]["argmin_target"].iloc[0])
        curve = fold_curve(rows, heldout, rate, base_rate)
        stab[label] = {
            "train_opt": sel,
            "at_edge": bool(sel >= rows["target"].max() - 1e-9),
            **stability(sel, curve, FINE),
        }

    report, stats = build_report(rows, base_rate, held, heldout, train, target, train_opt, stab)

    # held-out curves, and the walk-forward, appended below
    extra = ["## 4. The held-out curve", ""]
    for label, rate in hs.RATES.items():
        c = fold_curve(rows, heldout, rate, base_rate)
        c = c[c["target"].isin([0.90, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999])]
        pv = c.assign(scheme=c["form"] + " × " + c["granularity"]).pivot(
            index="target", columns="scheme", values="cost"
        )
        pv = pv[[f"{BASE[0]} × {BASE[1]}", f"{FINE[0]} × {FINE[1]}"]].reset_index()
        extra += [
            f"Fold {heldout} mean cost per fold, holding at {label}:",
            "",
            bt._markdown_table(pv, float_cols=tuple(pv.columns)),
            "",
        ]
    extra += ["## 5. Walk-forward (select on earlier folds, score the next)", ""]
    for label, rate in hs.RATES.items():
        wf = walk_forward(rows, base_rate, rate)
        extra += [
            f"Holding at {label} (matched-service selection; point estimates, one 28-day "
            "window each):",
            "",
            bt._markdown_table(
                wf, float_cols=("selected_target", "cost_diff", "csl_diff_pp", "fill_diff_pp")
            ),
            "",
        ]
    if target is not None:
        cols = None
        for name, fl in (
            (f"held-out fold {heldout}", [heldout]),
            ("all four folds (in-sample)", folds_all := [int(f) for f in folds]),
        ):
            d = decompose(rows, fl, target, base_rate)
            cols = tuple(c for c in d.columns if c != "holding rate")
            extra += [
                f"### Decomposition on {name}: scheme vs. target",
                "",
                bt._markdown_table(d.round(0), float_cols=()),
                "",
            ]
        extra += [
            "*The quintile-at-97% comparison bundles two changes. Where 'target effect' is most of "
            "'total', the saving is mostly the price of carrying more stock, not a better "
            "allocation of it — and that is exactly the part that depends on the holding rate.*",
            "",
        ]
        del cols, folds_all
    extra += verdict_section(stats, stab)
    out = bt.REPORTS_DIR / f"holdout_target_{dt.date.today().isoformat()}.md"
    out.write_text(report + "\n" + "\n".join(extra))
    print(f"Wrote {out}")
    print({k: v for k, v in stats.items() if k in ("verdicts", "csl_ok")})


if __name__ == "__main__":
    main()
