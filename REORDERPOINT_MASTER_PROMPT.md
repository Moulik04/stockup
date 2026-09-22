# ReorderPoint — Master Prompt for Claude Code

You are the lead engineer on **ReorderPoint**: an end-to-end demand forecasting system that ends in an inventory decision, with deployment and monitoring. The owner is MJ, a recent data science graduate building this as a flagship portfolio project for data scientist roles. MJ also runs a small family retail business with physical inventory, which gives this project a real second dataset and a real user.

Read this whole file before writing any code. Re-read **Hard constraints** and **Checkpoints** before every phase.

---

## 1. The one-line goal

> Given sales history for many products, forecast demand over the next horizon with calibrated uncertainty, then convert the forecast into a **reorder decision** (how much, when) that minimises stockout plus holding cost — and prove it beats naive baselines in honest backtests.

The output of the system is a reorder table, not a chart. Every modelling choice must be justified by its effect on the decision, not just on RMSE.

## 2. Hard constraints (non-negotiable)

1. **Zero cost.** No paid APIs, cloud accounts with billing, or paid tiers. If a step costs money, stop and propose the free alternative.
2. **No hosted LLM APIs anywhere in the pipeline.** This is a classical ML project; LLMs are not needed to build it.
3. **No leakage, ever.** All backtests use rolling-origin evaluation. Features are built only from data available at forecast time. Any feature that would not be known at prediction time is a bug.
4. **Baselines first.** No model is reported without being compared to seasonal naive and a simple moving average on the same folds.
5. **Two data tracks, one codebase.** Public data (Track A) is what lives in the repo and README. MJ's business data (Track B) is private, never committed, and runs through the exact same pipeline via a config switch.
6. **Reproducible.** Fresh clone + `make setup` + `make data` + `make backtest` runs on an Apple Silicon Mac with only Python installed.

## 3. Environment

- **Primary:** MacBook M2. Everything in Phases 0–6 must run here in reasonable time (LightGBM, statsforecast, pandas/polars).
- **Heavy compute:** PSC Bridges-2, NVIDIA V100 32 GB. Use only for Phase 7 (deep models: N-BEATS / TFT / PatchTST via `neuralforecast` or `darts`) and hyperparameter sweeps. V100: **fp16 only, no bf16, no FlashAttention.** Write `scripts/bridges2/*.sbatch` and `docs/bridges2.md`; verify module names with MJ or the PSC docs before hardcoding.
- Python 3.11+, `uv` or `pip` with lockfile. Prefer `polars` for the feature pipeline if data is large; otherwise pandas is fine.

## 4. Data

**Track A — public (in the repo):** M5 Forecasting dataset (Walmart daily unit sales, ~30k series, hierarchical: item × store × dept × category × state, with prices and calendar/SNAP events). Free via Kaggle (needs a free Kaggle account; `kaggle` CLI download script). Start with a **subset** (one state, or one category) so iteration is fast; scale up in Phase 5.

**Track B — private (MJ's business):** sales by SKU × month (or week) from the business. Loaded from a local CSV/Excel path set in `.env`, with a `data/track_b/README.md` describing the expected columns. Never committed. Expect strong annual seasonality (summer peak), fewer series, monthly grain, and possibly missing/zero-inflated months. The pipeline must handle both grains via config.

Write `docs/data.md`: schema, grain, horizon, what "demand" means (sales as a proxy for demand; note stockout censoring as a known limitation).

## 5. Architecture

```
raw data ──► ingest ──► canonical long table (series_id, date, y, price, exog...)
                              │
                              ▼
                     feature pipeline (lags, rolling stats, calendar,
                     price changes, event flags) — all point-in-time safe
                              │
                              ▼
        ┌───────────── model registry ─────────────┐
        │ seasonal naive │ ETS/Theta (statsforecast) │ global LightGBM (point + quantile) │ [deep, Phase 7] │
        └───────────────────────────────────────────┘
                              │
                              ▼
                 rolling-origin backtester → metrics + fold-level reports
                              │
                              ▼
                 quantile forecasts (P10/P50/P90) for horizon H
                              │
                              ▼
                 decision layer: safety stock, reorder point, order quantity,
                 cost simulation (stockout cost vs holding cost)
                              │
                              ▼
                 FastAPI (batch scoring + on-demand) ─┬─ Streamlit dashboard
                                                     └─ monitoring: accuracy drift, data drift, alerts
```

Modules under `reorderpoint/`: `ingest.py`, `features.py`, `models/` (one file per model family, common interface), `backtest.py`, `metrics.py`, `decision.py`, `serve.py`, `monitor.py`, `config.py`.

## 6. Modelling ladder (build in this order, keep every rung)

1. **Seasonal naive** and **moving average** — the floor.
2. **Statistical** — ETS, Theta, and optionally ARIMA via `statsforecast` (fast, free, strong baselines).
3. **Global gradient-boosted model** — one LightGBM model across all series with lag/rolling/calendar/price features. Train point (L2 or Tweedie for zero-inflated demand) and **quantile** models (P10, P50, P90).
4. **Hierarchical reconciliation** (Track A only) — bottom-up vs MinT via `hierarchicalforecast`. Show whether coherence helps.
5. **Deep (Phase 7, Bridges-2)** — N-BEATS or PatchTST. Only worth reporting if it beats LightGBM on the decision metric.

## 7. Evaluation

- **Backtest protocol:** rolling origin, ≥4 folds, horizon H = 28 days (Track A) / 3 months (Track B), gap of 0. Fold definitions live in one place and are reused by every model.
- **Accuracy metrics:** MASE and RMSSE (scale-free, comparable across series), WRMSSE for M5 comparability, and **pinball loss / coverage** for quantiles (a P90 that covers 70% is broken).
- **Decision metrics (the ones that matter):** simulated total cost = stockout cost + holding cost over the backtest, fill rate, and inventory turns — computed from the reorder decisions each model's forecast produces. This is the headline table.
- `make backtest` writes `reports/backtest_<date>.md` with a model × metric table, per-fold breakdown, and per-category winners. Include a "where the model fails" section (new items, intermittent demand, promo weeks).

## 8. Decision layer

- Inputs: quantile forecast, lead time, current on-hand stock, unit cost, holding cost rate, stockout penalty (all in config, USD).
- Outputs per SKU: reorder point, safety stock, recommended order quantity, expected stockout probability over lead time, and a one-line rationale.
- Implement a simple, explainable policy first (newsvendor-style service-level target from the quantile forecast). Document the formula in `docs/decision.md`. Fancier policies only if the simulation shows they pay off.

## 9. Deployment and monitoring

- **FastAPI**: `POST /forecast` (series ids, horizon), `POST /reorder` (adds stock levels, returns decisions), `GET /health`, `GET /metrics`.
- **Batch job**: `make score` produces `outputs/reorder_<date>.csv`.
- **Dockerfile** and a GitHub Actions workflow (free tier) that runs tests and a smoke backtest on a tiny fixture.
- **Monitoring** (`monitor.py`): once actuals arrive, log realised error per series, track rolling MASE, flag series whose error jumped, and detect input drift (distribution shift in recent sales vs training window). Streamlit page shows it. This is Phase 6 and is what turns "a model" into "a system".
- Streamlit dashboard: pick a SKU → history, forecast fan chart, reorder recommendation, backtest performance, monitoring status.

## 10. Phases and acceptance criteria

Work strictly in order; do not start a phase until the previous one's criteria pass and MJ signs off.

**Phase 0 — Scaffold (½ day).** Repo layout, `pyproject.toml`, `Makefile`, ruff/black pre-commit, pytest skeleton, `DECISIONS.md`, README stub, `.env.example`, MIT licence. `make setup && make test` passes.

**Phase 1 — Data (1 day).** Kaggle download script, ingest to canonical long table (parquet), Track B loader with the same output schema, `docs/data.md`, EDA notebook (seasonality, intermittency, price effects) exported to `reports/eda.md`. Acceptance: `make data` works on both tracks; schema tests pass.

**Phase 2 — Baselines + backtester (1 day).** Rolling-origin backtester, seasonal naive, moving average, statsforecast models, metrics module. Acceptance: `make backtest` produces a report; baseline numbers recorded. These are the "before" numbers in the README.

**Phase 3 — Global LightGBM (2 days).** Point-in-time-safe feature pipeline, point + quantile models, leakage tests (a test that shifts the target and confirms features don't move). Acceptance: beats seasonal naive on MASE across ≥3 of 4 folds; quantile coverage within ±5 points of nominal.

**Phase 4 — Decision layer (1 day).** Reorder policy, cost simulation, headline decision-metric table across all models. Acceptance: the best model's simulated cost is lower than seasonal naive's, and the report shows by how much in USD.

**Phase 5 — Scale + reconciliation (1 day).** Full M5 (or as large as the M2 handles), hierarchical reconciliation experiment, per-category analysis. Acceptance: report answers "does reconciliation help, and where?"

**Phase 6 — Serve + monitor (1–2 days).** FastAPI, batch scoring, Dockerfile, CI, monitoring module, Streamlit dashboard. Acceptance: docker container answers `/reorder` correctly; CI green; monitoring flags a synthetically drifted series in a test.

**Phase 7 — Deep models on Bridges-2 (stretch, 1–2 days).** neuralforecast N-BEATS/PatchTST, fp16, SLURM. Acceptance: honest comparison in the report; keep only if it wins on decision cost.

**Phase 8 — Portfolio polish (1 day).** README: problem, data, architecture diagram, headline table (accuracy + cost), ablations, "where it fails", Track B case study (aggregated results only, no raw business data), 60-second demo GIF, "what I'd do with a budget" section. Tag `v1.0.0`.

## 11. Engineering conventions

- Typed, small, pure functions; every model implements the same `fit / predict_quantiles` interface so the backtester treats them identically.
- Tests alongside code; a leakage test is mandatory before any feature ships.
- Conventional commits; never commit data, models, or `.env`.
- `DECISIONS.md`: context → options → decision → consequence, for every non-obvious choice.
- Boring tools over clever ones; no dependency for what 30 lines of Python does.
- Write for a hiring manager skimming the repo: the README's first three lines answer "what problem, what result, why should I care?"
- Sharp naming matters to MJ. Propose names; no `utils.py`, no `helpers.py`.

## 12. Checkpoints — stop and ask MJ

- Before installing any tool not named here.
- Before anything touching Bridges-2.
- Before changing the canonical data schema after Phase 1.
- Before setting cost parameters for Track B (MJ knows the real numbers).
- When a free option is clearly worse and MJ should know the trade-off.
- At the end of every phase: what was built, the metrics, what you'd change, one question for MJ.

## 13. Definition of done

A stranger can clone the repo, run three commands, and get a backtest report showing the model beats seasonal naive on both accuracy and simulated inventory cost, then start the API and get a reorder recommendation with a P10/P90 band and a rationale for any SKU. The README shows the before/after numbers and a monitoring view. Separately, MJ can point the same pipeline at the business data and get a reorder table for next season.

## 14. First action

Confirm you have read this file. Propose the Phase 0 repo tree, list assumptions about MJ's machine (Python version, Kaggle CLI, Docker installed or not), and ask MJ whether Track B data is available now or later. Wait for the go-ahead before creating files.
