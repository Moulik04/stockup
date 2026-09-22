"""v1.1 Task 3 / Task 9: paired-bootstrap CIs on MinTrace-vs-base (and BottomUp-vs-base) MASE, per
hierarchy level.

The reconciliation experiment reports mean MASE per level, and the README quotes MinTrace's
-0.10% / -0.12% at the two finest levels as a real effect. Those are differences between two
forecasts of the *same nodes*, so the bootstrap resamples nodes and recomputes both sides on the
same draw.

**A level needs enough nodes for this to mean anything.** The hierarchy has 2, 2, 6, 14 and 400
nodes per level. Resampling two states with replacement can only ever produce three distinct
samples; that is not inference. Levels below `MIN_NODES` are reported as point estimates only,
with the reason stated, rather than given an interval that looks more informative than it is —
which also means the README's upper-level claims (BottomUp +3.6% at state, MinTrace +1.1%) cannot
be tested by this method at all.

Reads `reconciliation_node_detail.parquet`, written by `make reconcile`.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from reorderpoint import backtest as bt
from reorderpoint import bootstrap as boot
from reorderpoint import reconcile as rc

MIN_NODES = 30
METHODS = ["MinTrace", "BottomUp"]


def level_comparisons(nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in rc.LEVEL_NAMES:
        lv = nodes[nodes["level"] == level]
        n_nodes = lv["series_id"].nunique()
        base = lv[lv["method"] == "base"]
        # nodes with a finite MASE under every method and fold, so all sides cover the same units
        ok = lv.groupby("series_id")["mase"].apply(lambda s: s.notna().all())
        keep = ok[ok].index
        base_mean = base[base["series_id"].isin(keep)].groupby("fold")["mase"].mean().mean()
        for method in METHODS:
            alt = lv[lv["method"] == method]
            row = {"level": level, "method": method, "n_nodes": n_nodes, "base_mase": base_mean}
            a = alt[alt["series_id"].isin(keep)]
            b = base[base["series_id"].isin(keep)]
            point = a.groupby("fold")["mase"].mean().mean() - base_mean
            if len(keep) >= MIN_NODES:
                # scale so the resampled per-fold *sum* over n draws equals the per-fold mean
                a = a.assign(mase=a["mase"] / len(keep))
                b = b.assign(mase=b["mase"] / len(keep))
                out = boot.bootstrap_diff(a, b, value_col="mase", series_col="series_id")
                row |= {k: out[k] for k in ("point", "ci_lo", "ci_hi", "crosses_zero")}
                row["frac_sign_flip"] = out["frac_sign_flip"]
                row["testable"] = True
            else:
                row |= {
                    "point": point,
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "crosses_zero": None,
                    "frac_sign_flip": float("nan"),
                    "testable": False,
                }
            rows.append(row)
    return pd.DataFrame(rows)


def build_report(table: pd.DataFrame) -> str:
    def pct(x: float, base: float) -> str:
        return "—" if pd.isna(x) else f"{x / base * 100:+.2f}%"

    show = table.assign(
        diff_pct=lambda d: [pct(x, b) for x, b in zip(d["point"], d["base_mase"], strict=True)],
        ci_pct=lambda d: [
            (
                "not testable (too few nodes)"
                if not t
                else (
                    "identical to base by construction (BottomUp is a no-op at the bottom level)"
                    if lo == hi == 0
                    else f"[{pct(lo, b)}, {pct(hi, b)}]" + (" — crosses zero" if c else "")
                )
            )
            for t, lo, hi, c, b in zip(
                d["testable"],
                d["ci_lo"],
                d["ci_hi"],
                d["crosses_zero"],
                d["base_mase"],
                strict=True,
            )
        ],
    )[["level", "method", "n_nodes", "base_mase", "point", "diff_pct", "ci_pct", "frac_sign_flip"]]
    show = show.rename(columns={"point": "mase_diff", "frac_sign_flip": "sign_flips"})

    fine = table[(table["level"] == rc.LEVEL_NAMES[-1]) & (table["method"] == "MinTrace")].iloc[0]
    fine_line = (
        f"At the bottom level (item × store, {fine['n_nodes']} nodes) MinTrace's MASE change is "
        f"{fine['point']:+.4f} (95% CI {fine['ci_lo']:+.4f} to {fine['ci_hi']:+.4f}) — "
        + (
            "**not distinguishable from zero.** The README's `-0.12%` is inside the noise and "
            "should be reported as such, not as a real improvement."
            if fine["crosses_zero"]
            else "**distinguishable from zero**, though tiny in size."
        )
    )

    return "\n".join(
        [
            f"# Reconciliation confidence intervals — {dt.date.today().isoformat()}",
            "",
            "Paired bootstrap over hierarchy nodes (2,000 resamples, seed 0) on mean MASE, "
            "MinTrace and BottomUp vs the unreconciled AutoETS base, per level. Base model is "
            "AutoETS "
            "(not the production LightGBM — its price/calendar features have no definition at a "
            "synthetic aggregate node), so this experiment says nothing about the production "
            "model regardless of what the intervals show.",
            "",
            bt._markdown_table(show, float_cols=("base_mase", "mase_diff", "sign_flips")),
            "",
            f"Levels with fewer than {MIN_NODES} nodes are not bootstrapped: resampling 2 states "
            "or 6 categories with replacement is not inference, and an interval there would look "
            "more informative than it is. Those rows are point estimates only, and the README's "
            "claims about them (BottomUp worse at every aggregate level; MinTrace roughly neutral "
            "above the item level) cannot be tested by this method.",
            "",
            fine_line,
            "",
        ]
    )


def main() -> None:
    nodes = pd.read_parquet(rc.NODE_DETAIL_PATH)
    table = level_comparisons(nodes)
    out = bt.REPORTS_DIR / f"reconciliation_ci_{dt.date.today().isoformat()}.md"
    out.write_text(build_report(table))
    print(f"Wrote {out}")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
