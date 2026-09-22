"""v1.1 Task 3: paired-bootstrap CIs on the model-vs-model comparisons the README ranks on.

LightGBM vs SeasonalNaive on cost is the headline claim; LightGBM vs each statistical baseline is
the one that "may not survive". Each is a difference between two models scored on the *same*
series and folds, so the bootstrap resamples series and recomputes both sides on every draw (see
`bootstrap.py`) — fold and series effects cancel in the difference.

Everything is read from caches an earlier run already paid for: per-series cost from
`decision_detail.parquet` (`make decide`), per-row forecasts from `backtest_forecasts.parquet`
(`make divergence`) for per-series MASE. No model is refit.

**Not covered here, and why** (also stated in the report):
- NBEATS vs LightGBM: NBEATS ran on Bridges-2 and only its aggregate tables came back; there is no
  per-series detail to resample. `scripts/bridges2/run_deep_backtest.py` now persists it, but a
  rerun is a GPU job and needs MJ's sign-off first.
- MinTrace vs base: see `reconcile.py` / `reports/reconciliation_ci_<date>.md`.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import decision as dec
from reorderpoint import divergence as dv

FOCUS = "LightGBM"
OTHERS = ["SeasonalNaive", "AutoETS", "AutoTheta", "MovingAverage"]


def per_series_mase(forecasts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, fold, series): MASE of the P50 forecast, scaled by the series' own
    in-sample seasonal-naive MAE over that fold's training window — the same definition
    `backtest.evaluate_fold` reports, recomputed from the cached forecasts."""
    eval_panel = bt.sample_series(panel, bt.N_SERIES_SAMPLE)
    frames = []
    for fold in bt.make_folds(eval_panel):
        train = eval_panel[eval_panel["date"] <= fold.train_end]
        scales = bt._in_sample_scales(train, bt.SEASON_LENGTH)["scale_mae"]
        fc = forecasts[forecasts["fold"] == fold.index].copy()
        fc["abs_err"] = (fc["y"] - fc["p50"]).abs()
        mae = fc.groupby(["model", "series_id"])["abs_err"].mean().reset_index()
        mae["scale"] = mae["series_id"].map(scales)
        mae["mase"] = mae["abs_err"] / mae["scale"]
        mae["fold"] = fold.index
        frames.append(mae[["model", "fold", "series_id", "mase"]])
    return pd.concat(frames, ignore_index=True)


def compare(detail: pd.DataFrame, metric: str, a: str, b: str, kind: str = "sum", **cols) -> dict:
    """CI on metric(a) - metric(b) across `detail`'s model column."""
    da, db = detail[detail["model"] == a], detail[detail["model"] == b]
    if kind == "ratio":
        return boot.bootstrap_ratio_diff(da, db, cols["num"], cols["den"], group_cols=["fold"])
    return boot.bootstrap_diff(da, db, value_col=metric, group_cols=["fold"])


def build_comparisons(cost_detail: pd.DataFrame, mase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    csl = cost_detail[cost_detail["policy"] == "csl"].assign(
        cycles_ok=lambda d: d["cycles"] - d["stockout_cycles"]
    )

    # keep only series with a finite MASE for every model x fold, so both sides of every draw
    # cover the same units and the resample sum equals n x the mean
    ok = mase.groupby("series_id")["mase"].apply(lambda s: s.notna().all())
    keep = ok[ok].index
    m = mase[mase["series_id"].isin(keep)].copy()
    m["mase"] = m["mase"] / len(keep)  # resampled sum over n draws == resampled mean

    for other in OTHERS:
        rows.append(
            {
                "comparison": f"{FOCUS} − {other}",
                "metric": "cost per fold ($)",
                **compare(csl, "total_cost", FOCUS, other),
            }
        )
    for other in OTHERS:
        rows.append(
            {
                "comparison": f"{FOCUS} − {other}",
                "metric": "fill rate (pp)",
                **{
                    k: (v * 100 if k in {"point", "ci_lo", "ci_hi"} else v)
                    for k, v in compare(
                        csl,
                        "",
                        FOCUS,
                        other,
                        kind="ratio",
                        num="units_shipped",
                        den="units_demanded",
                    ).items()
                },
            }
        )
    for other in OTHERS:
        rows.append(
            {
                "comparison": f"{FOCUS} − {other}",
                "metric": "MASE",
                **compare(m, "mase", FOCUS, other),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_mase_series"] = len(keep)
    return out


def build_report(table: pd.DataFrame) -> str:
    def verdict(r: pd.Series) -> str:
        if r["crosses_zero"]:
            return "not distinguishable at this sample size"
        # lower is better for cost and MASE, higher for fill rate
        better = r["point"] > 0 if r["metric"] == "fill rate (pp)" else r["point"] < 0
        return f"{FOCUS} better" if better else f"{FOCUS} worse"

    show = table.assign(verdict=table.apply(verdict, axis=1))[
        [
            "comparison",
            "metric",
            "point",
            "ci_lo",
            "ci_hi",
            "frac_sign_flip",
            "verdict",
        ]
    ].rename(columns={"point": "diff", "frac_sign_flip": "sign_flips"})
    n_series = table.attrs.get("n_mase_series", "?")

    cost = table[table["metric"] == "cost per fold ($)"].set_index("comparison")
    survives = [c.split(" − ")[1] for c, r in cost.iterrows() if not r["crosses_zero"]]
    fails = [c.split(" − ")[1] for c, r in cost.iterrows() if r["crosses_zero"]]
    mase = table[table["metric"] == "MASE"].set_index("comparison")
    mase_worse = [
        c.split(" − ")[1] for c, r in mase.iterrows() if not r["crosses_zero"] and r["point"] > 0
    ]
    mase_tie = [c.split(" − ")[1] for c, r in mase.iterrows() if r["crosses_zero"]]

    lines = [
        f"# Model-vs-model confidence intervals — {dt.date.today().isoformat()}",
        "",
        "Paired bootstrap over series (2,000 resamples, seed 0): each draw resamples the 400 "
        "series with replacement and recomputes *both* models on the same draw, so the interval "
        "is on the difference. Same 400-series / 4-fold sample as every report; CSL policy "
        "(`z · σ_pooled`, 2 intermittency buckets — the shipped scheme). `diff` is "
        f"{FOCUS} minus the other model; `sign_flips` is the share of resamples whose sign "
        "disagrees with the point estimate.",
        "",
        bt._markdown_table(show, float_cols=("diff", "ci_lo", "ci_hi", "sign_flips")),
        "",
        f"MASE rows use the {n_series} series with a finite scale in every fold (the rest have "
        "an all-constant training window); cost and fill-rate rows use all series.",
        "",
        "## What survives",
        "",
        f"**Cost.** {FOCUS}'s cost advantage is distinguishable from zero against "
        f"{', '.join(survives) if survives else 'none of the baselines'}"
        + (f", and **not distinguishable** against {', '.join(fails)}." if fails else "."),
        "",
        f"**MASE.** {FOCUS} is distinguishably *worse* than {', '.join(mase_worse) or 'none'}"
        + (f" and not distinguishable from {', '.join(mase_tie)}." if mase_tie else "."),
        "",
        "Any comparison marked *not distinguishable* is a statement about this sample, not about "
        "the models: with 400 series and four folds the gaps between the four clustered models "
        "(see `reports/model_divergence_<date>.md`) are smaller than the noise. It should be "
        "reported as inconclusive rather than as a ranking.",
        "",
        "## Not covered",
        "",
        "- **NBEATS vs LightGBM** (accuracy and cost): NBEATS was trained on Bridges-2 and only "
        "aggregate tables came back, so there is no per-series detail to resample. "
        "`scripts/bridges2/run_deep_backtest.py` now saves it, but recovering it needs a GPU "
        "rerun, which needs MJ's approval. Until then the README's NBEATS numbers remain point "
        "estimates — the 18%-worse-on-cost gap is large enough that it is unlikely to be noise, "
        "but that is an inference, not a measurement.",
        "- **MinTrace vs base**: see `reports/reconciliation_ci_<date>.md`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    panel = bt.load_eval_panel()
    cost_detail = pd.read_parquet(dec.DETAIL_PATH)
    forecasts = pd.read_parquet(dv.FORECASTS_PATH)

    mase = per_series_mase(forecasts, panel)
    table = build_comparisons(cost_detail, mase)
    # sanity: the recomputed MASE must reproduce the published per-model means
    kept = mase.groupby("model")["mase"].mean()
    print("recomputed mean MASE per model (all finite series):")
    print(kept.round(3).to_string())

    out = bt.REPORTS_DIR / f"model_comparison_ci_{dt.date.today().isoformat()}.md"
    out.write_text(build_report(table))
    print(f"Wrote {out}")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
