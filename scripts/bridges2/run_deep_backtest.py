"""Runs NBEATS through the SAME backtest + decision-cost harness every other model already went
through (backtest.py / decision.py — unchanged), for a direct, honest comparison. Meant to run on
a Bridges-2 GPU node (see docs/bridges2.md), not the M2.

Only SeasonalNaive (the mandatory floor, constraint #4) and NBEATS are actually recomputed here.
AutoETS/AutoTheta/LightGBM are NOT re-run: same panel, same N_SERIES_SAMPLE/SAMPLE_SEED/HORIZON/
N_FOLDS as Phase 2-4 (all fixed module constants, untouched), so their existing numbers in
reports/backtest_2026-09-03.md and reports/decision_2026-09-05.md are already exactly what a
rerun would produce — recomputing them here would just spend this account's limited GPU-hour
balance on models that don't need a GPU at all.

**Queued for this run (2026-09-21): NBEATS under `S = s + 2 x lead-time demand`, not just the
original `S = s`.** The 2026-09-10 result (NBEATS wins on accuracy, loses badly on decision cost)
was measured entirely under `S = s`, the policy `order_up_to.py` later showed was defective (78%
of stockout cycles at 95% were cycles the reorder point would have covered — the order never
refills past s, so demand mid-lead-time goes uncovered until the next cycle). The root-cause
hypothesis for NBEATS's loss was that seasonal naive's inaccuracy functioned as an unpriced safety
margin NBEATS's superior accuracy doesn't carry — but `S = s` denies every model a *priced* buffer
too, so NBEATS may have been penalised twice: once for lacking naive's accidental cushion, and
again by a policy that had no real buffer for the leanest model to fall back on. `S > s` adds a
buffer directly, independent of forecast bias. If the root-cause story is right, NBEATS should
close some of its gap to LightGBM under `S = s + 2 x lead-time demand` (LightGBM's own chosen lot,
see `reports/order_up_to_2026-09-20.md`); if it doesn't, NBEATS's loss has a separate cause the
synthesis should say so rather than paper over. Either outcome is informative — this is not a run
expected to vindicate NBEATS, it's a run that disambiguates two competing explanations for why it
lost. `decision.run_decision_backtest` now takes `lot_multiples`, so NBEATS is fit once per fold
and both order-up-to levels are resimulated off that one fit — this costs no extra GPU time over
the original `S = s`-only run, only the (CPU-side) resimulation. Writes both
`decision_deep_<date>.md` (S = s, for continuity with the 2026-09-10 numbers) and
`decision_deep_lot2_<date>.md` (S = s + 2 x lead-time demand); compare them directly rather than
either alone.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# Run via `python3 scripts/bridges2/run_deep_backtest.py` (a direct script path, not `-m`), which
# does not add the repo root to sys.path the way `-m`/`-c` do. Standalone utility scripts avoid
# importing reorderpoint for this reason, but this script genuinely needs the package, so it
# fixes up sys.path itself instead.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from reorderpoint import backtest as bt
from reorderpoint import decision
from reorderpoint.config import load_config
from reorderpoint.models import deep, naive

bt.MODEL_FACTORIES = {
    "SeasonalNaive": naive.seasonal_naive,
    "NBEATS": deep.nbeats_global,
}


def main() -> None:
    panel = pd.read_parquet(bt.PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    date_str = dt.date.today().isoformat()
    bt.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running accuracy backtest (SeasonalNaive + NBEATS)...")
    results, detail = bt.run_backtest(panel)
    # Per-series detail is what a paired bootstrap over series needs to put a CI on NBEATS vs
    # LightGBM (reorderpoint/model_ci.py). It used to be discarded here, so only aggregate tables
    # came back from the cluster and the comparison could not be bootstrapped afterwards.
    detail.to_parquet(bt.PANEL_PATH.parent / "deep_backtest_detail.parquet", index=False)
    dept_lookup = panel.drop_duplicates("series_id").set_index("series_id")["dept_id"]
    report = bt.write_report(results, detail, dept_lookup)
    backtest_path = bt.REPORTS_DIR / f"backtest_deep_{date_str}.md"
    backtest_path.write_text(report)
    print(f"Wrote {backtest_path}")

    # Queued 2026-09-21: S = s (for continuity with the 2026-09-10 numbers) and
    # S = s + 2 x lead-time demand (the policy that replaced it in production, LightGBM's own
    # chosen lot — reports/order_up_to_2026-09-20.md) in ONE call, so NBEATS is fit once per fold
    # and both order-up-to levels are resimulated from that single fit — see the module docstring
    # for the hypothesis this comparison is meant to test.
    print("Running decision-cost backtest (SeasonalNaive + NBEATS), S = s and S = s + 2x LT...")
    config = load_config()
    decision_results, decision_detail = decision.run_decision_backtest(
        panel, config.costs, lot_multiples=(0.0, 2.0)
    )
    decision_detail.to_parquet(bt.PANEL_PATH.parent / "deep_decision_detail.parquet", index=False)

    for lot, suffix in ((0.0, ""), (2.0, "_lot2")):
        results_lot = decision_results[decision_results["lot_multiple"] == lot]
        report = decision.write_decision_report(results_lot, config.costs)
        path = bt.REPORTS_DIR / f"decision_deep{suffix}_{date_str}.md"
        path.write_text(report)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
