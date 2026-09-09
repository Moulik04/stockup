"""Runs NBEATS through the SAME backtest + decision-cost harness every other model already went
through (backtest.py / decision.py — unchanged), for a direct, honest comparison. Meant to run on
a Bridges-2 GPU node (see docs/bridges2.md), not the M2.

Only SeasonalNaive (the mandatory floor, constraint #4) and NBEATS are actually recomputed here.
AutoETS/AutoTheta/LightGBM are NOT re-run: same panel, same N_SERIES_SAMPLE/SAMPLE_SEED/HORIZON/
N_FOLDS as Phase 2-4 (all fixed module constants, untouched), so their existing numbers in
reports/backtest_2026-09-03.md and reports/decision_2026-09-05.md are already exactly what a
rerun would produce — recomputing them here would just spend this account's limited GPU-hour
balance on models that don't need a GPU at all.
"""

from __future__ import annotations

import datetime as dt

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
    dept_lookup = panel.drop_duplicates("series_id").set_index("series_id")["dept_id"]
    report = bt.write_report(results, detail, dept_lookup)
    backtest_path = bt.REPORTS_DIR / f"backtest_deep_{date_str}.md"
    backtest_path.write_text(report)
    print(f"Wrote {backtest_path}")

    print("Running decision-cost backtest (SeasonalNaive + NBEATS)...")
    config = load_config()
    decision_results, _decision_detail = decision.run_decision_backtest(panel, config.costs)
    decision_report = decision.write_decision_report(decision_results, config.costs)
    decision_path = bt.REPORTS_DIR / f"decision_deep_{date_str}.md"
    decision_path.write_text(decision_report)
    print(f"Wrote {decision_path}")


if __name__ == "__main__":
    main()
