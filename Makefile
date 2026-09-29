.PHONY: setup data data-b data-full data-reconcile eda backtest divergence decide safety-stock service-level holding-sensitivity breakeven optimal-target penalty holdout croston production-model reconcile train calibrate score serve dashboard test lint format

setup:
	uv sync
	uv run pre-commit install

data:
	uv run python scripts/download_m5.py
	uv run python -m reorderpoint.ingest

# Track B: download UCI Online Retail II (CC BY 4.0), clean it to the weekly SKU panel, and write
# reports/track_b_cleaning_<date>.md (what every rule removed, zero share, seasonality, bulk orders).
data-b:
	python3 scripts/download_online_retail.py
	uv run python -m reorderpoint.online_retail

# Full M5 (all 3 categories) — needed for reconcile's cross-category hierarchy. Not the
# `data` default: HOBBIES-only keeps Phase 0-4's day-to-day iteration fast. Safe on an 8.6GB
# machine because every full-scale reader uses predicate-pushdown reads, not a full in-memory load.
data-full:
	uv run python scripts/download_m5.py
	uv run python -m reorderpoint.ingest --cat-ids HOBBIES HOUSEHOLD FOODS

# The slice Phase 5's reconciliation runs on (stores CA_1/TX_1, all three categories), written
# beside — not over — the HOBBIES panel every other target uses.
data-reconcile:
	uv run python -m reorderpoint.ingest --store-ids CA_1 TX_1 --cat-ids HOBBIES HOUSEHOLD FOODS --out data/track_a/processed/panel_reconcile.parquet

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
	uv run jupyter nbconvert --to markdown notebooks/eda.ipynb --output-dir reports --output eda

backtest:
	uv run python -m reorderpoint.backtest

# Reruns the backtest's forecasts once and caches them (data/track_a/processed/backtest_forecasts.parquet)
divergence:
	uv run python -m reorderpoint.divergence

decide:
	uv run python -m reorderpoint.decision

# Safety-stock calibration ablation: 2x2 of {normal, empirical} x {intermittency,
# volume_tercile_within_intermittency} plus a volume_quintile cell, with bootstrap CIs.
# Reuses the forecast cache from `make divergence`, so it costs simulations, not backtests.
safety-stock:
	uv run python -m reorderpoint.ablate_safety_stock

# Cost-optimal service level: critical ratio + cost-vs-target sweep (0.80-0.999).
service-level:
	uv run python -m reorderpoint.service_level_sweep

# Re-prices the stored runs at four readings of `holding_cost_rate` (2%/day, 2%/month, 2%/year, 25%/yr):
# does the model cost ranking, the calibration-scheme win, or the optimal service level move?
holding-sensitivity:
	uv run python -m reorderpoint.holding_sensitivity

# Solves for the holding rate at which LightGBM and SeasonalNaive cost the same (~364%/yr), checks it
# three independent ways, and writes the cost-vs-rate chart the README and dashboard are built on.
breakeven:
	uv run python -m reorderpoint.holding_breakeven

# Extends the service-target grid to 99.9999%, reports the critical ratio and the cost-optimal
# target, and re-runs the model comparison there with bootstrap CIs (reports/optimal_target_*.md).
optimal-target:
	uv run python -m reorderpoint.optimal_target

# Ranking sensitivity to the stockout penalty (lost margin) and the service target
# (reports/penalty_sensitivity_*.md). Run `make optimal-target` first.
penalty:
	uv run python -m reorderpoint.penalty_sensitivity

# Picks the service-level target on folds 1-3 only and scores it on the held-out final fold.
holdout:
	uv run python -m reorderpoint.holdout_target

# The Croston family (CrostonClassic, CrostonSBA, TSB) — added to the model registry, scored on
# accuracy and, under the shipped S = s + 2 x lead-time demand policy, on decision cost, per
# intermittency bucket with bootstrap CIs (reports/croston_*.md). Extends the forecast/calibration
# caches with just these three models; the five already there are not re-fit.
croston:
	uv run python -m reorderpoint.croston

# Is LightGBM still the right model to serve? Runs the shipped policy with every model's P10/P90
# scrambled (do the native quantiles feed sizing?), paired-bootstrap cost of each cluster member
# and the Croston family against LightGBM, fit time / interval coverage, the calibrated interval
# `/forecast` shows against the native one, and that the served trailing mean forecasts what the
# ladder's MovingAverage does (reports/production_model_*.md). Reuses the caches `make croston`
# extends. Re-times every model, AutoTheta included: several minutes.
production-model:
	uv run python -m reorderpoint.production_model

reconcile:
	uv run python -m reorderpoint.reconcile
	uv run python -m reorderpoint.reconcile_ci

train:
	uv run python -m reorderpoint.train

# Refit only the safety-stock calibration `make serve` sizes with (models/production/
# safety_stock_calibration.joblib), leaving the persisted model as is. `make train` does both.
calibrate:
	uv run python -m reorderpoint.train --calibration-only

score:
	uv run python -m reorderpoint.serve --batch

serve:
	uv run uvicorn reorderpoint.serve:app --reload

dashboard:
	uv run streamlit run reorderpoint/dashboard.py

test:
	uv run python -m pytest

lint:
	uv run python -m ruff check .
	uv run python -m black --check .

format:
	uv run python -m ruff check --fix .
	uv run python -m black .
