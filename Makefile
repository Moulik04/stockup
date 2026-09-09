.PHONY: setup data data-full eda backtest decide reconcile train score serve dashboard test lint format

setup:
	uv sync
	uv run pre-commit install

data:
	uv run python scripts/download_m5.py
	uv run python -m reorderpoint.ingest

# Full M5 (all 3 categories) — needed for reconcile's cross-category hierarchy. Not the
# `data` default: HOBBIES-only keeps Phase 0-4's day-to-day iteration fast. Safe on an 8.6GB
# machine because every full-scale reader uses predicate-pushdown reads, not a full in-memory load.
data-full:
	uv run python scripts/download_m5.py
	uv run python -m reorderpoint.ingest --cat-ids HOBBIES HOUSEHOLD FOODS

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
	uv run jupyter nbconvert --to markdown notebooks/eda.ipynb --output-dir reports --output eda

backtest:
	uv run python -m reorderpoint.backtest

decide:
	uv run python -m reorderpoint.decision

reconcile:
	uv run python -m reorderpoint.reconcile

train:
	uv run python -m reorderpoint.train

score:
	uv run python -m reorderpoint.serve --batch

serve:
	uv run uvicorn reorderpoint.serve:app --reload

dashboard:
	uv run streamlit run reorderpoint/dashboard.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run ruff check --fix .
	uv run black .
