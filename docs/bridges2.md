# PSC Bridges-2 (Phase 7, stretch)

Deep model (N-BEATS via `neuralforecast`) on Bridges-2's V100 GPUs, fp16 only — V100 (Volta)
doesn't support bf16 tensor cores or FlashAttention's fused kernels properly, so both are
explicitly avoided. Everything below is verified against the real cluster (account `$USER`,
allocation `$PSC_ALLOCATION`, GPU balance ~481/500 SU), not assumed from generic docs.

## Access

Login: `bridges2.psc.edu`, or the [Open OnDemand web portal](https://ondemand.bridges2.psc.edu/)
(has a browser-based shell under Clusters — no local SSH key needed). Password auth via PSC
credentials (set at apr.psc.edu); this machine has no SSH keypair registered with PSC's
[key manager](http://grants.psc.edu/cgi-bin/ssh/listKeys.pl), so interactive password login is
the path in use.

## Storage

- `$HOME` = `/jet/home/$USER`, on `/jet` (347T total, well under half used) — effectively
  unconstrained.
- `/ocean/projects/$PSC_ALLOCATION/$USER/` — the project allocation, 10GB, almost entirely free
  (confirmed via `df`: 40K used). The OnDemand balance banner shows *remaining*, not used — easy
  to misread as "9.9/10.0 GB used" when it means the opposite.
- `$LOCAL` — node-local scratch, wiped after each job. Fine for job-scoped temp files, wrong
  place for anything meant to persist across job submissions (see venv setup below).

## GPU partitions

`GPU` (full-node) and `GPU-shared` (fractional) both exist. Nodes are heterogeneous hardware —
confirmed via `sinfo -N -p GPU-shared -o "%N %G"`:

- `v0xx` nodes: `gpu:v100-32:8` — the V100 32GB nodes this project targets.
- `gl0xx` nodes: `gpu:l40s-48:8` — L40S, a different generation, not used here.
- Partition-wide GRES total: `gres/gpu:v100-16=72, gres/gpu:v100-32=200, gres/gpu:l40s-48=24`
  (`scontrol show partition GPU-shared`) — note `v100-16` (16GB) is a separate pool from
  `v100-32`; the sbatch scripts request `--gres=gpu:v100-32:1` specifically, not bare `gpu:1`.

## Python environment

`module load pytorch/26.05-2.11-py3` gives Python 3.13.7, PyTorch 2.11.0+cu126, and a long list of
other packages — but it's a **read-only** `uv`-managed environment (confirmed: `uv pip install`
into it fails with `EACCES`). The module's own docs suggest cloning it via `cd $LOCAL; uv venv
...`, but `$LOCAL` doesn't survive between job submissions, so `scripts/bridges2/setup_env.sh`
does the same clone into `/ocean/projects/$PSC_ALLOCATION/$USER/stockup-env` instead — built once,
reused by every later job.

**One-time setup** (from a Bridges-2 shell — OnDemand web shell or `ssh`):

```bash
git clone https://github.com/Moulik04/stockup.git   # or `git pull` if already cloned
cd stockup
bash scripts/bridges2/setup_env.sh
```

## Verifying the neuralforecast API before integrating it

`neuralforecast`'s exact `NBEATS` constructor kwargs and `predict()` output column names weren't
assumed — `scripts/bridges2/diagnose_nbeats.py` (submitted via `diagnose.sbatch`) trained a tiny
NBEATS on synthetic data on a real V100 (`v018`) and confirmed, against the actually-installed
`neuralforecast==3.2.2`:

- **fp16 really runs, not bf16**: log shows `Using 16bit Automatic Mixed Precision (AMP)` with
  `precision="16-mixed"` passed through `**trainer_kwargs` (PyTorch Lightning under the hood).
- **`predict()` output columns**: `{alias}-lo-80.0`, `{alias}-median`, `{alias}-hi-80.0` for
  `loss=MQLoss(quantiles=[0.1, 0.5, 0.9])` — Nixtla's shared lo/hi/level convention (same family
  as the statsforecast wrapper in `models/base.py`), with a `.0` suffix on the level number and a
  `-median` column instead of the bare alias for P50. Not the `-q-10/50/90` naming that would've
  been an equally plausible guess.
- **NBEATS forecasts the full horizon in one `predict()` call** — no recursive rollout needed
  (unlike `models/lightgbm_global.py`), since `predict()` continues forward from wherever `fit()`
  ended.

`reorderpoint/models/deep.py` implements the real integration from these confirmed facts.
`neuralforecast`/`torch` are never imported by `backtest.py` directly — `deep.py` is only ever
imported by `scripts/bridges2/run_deep_backtest.py`, which runs on Bridges-2, not the M2 (verified
locally: `make test`/`make lint` pass unaffected on this machine, which doesn't have `torch`
installed at all).

## Scope: univariate, no exogenous features

Unlike LightGBM's feature pipeline (lags, rolling stats, calendar, price, SNAP), the NBEATS
integration here only uses each series' own `y` history — no `hist_exog_list`/`futr_exog_list`
passed to the model. This is a deliberate Phase 7 scoping choice: extending N-BEATS to consume the
same exogenous features LightGBM does is a real feature-engineering project of its own, not
something a stretch phase should absorb. The comparison in `reports/backtest_deep_<date>.md` is
still an honest apples-to-apples accuracy/cost comparison — it just means a NBEATS loss (if any)
against LightGBM could partly reflect this feature gap, not only architecture.

## Running the real comparison

`scripts/bridges2/run_deep_backtest.py` reuses `backtest.py`/`decision.py`'s existing
fold/sampling/report machinery unchanged (monkeypatches `bt.MODEL_FACTORIES` down to just
`SeasonalNaive` + `NBEATS` before calling `bt.run_backtest`/`decision.run_decision_backtest`) —
same 400-series sample, same 4 folds, same seed as every earlier phase, so the results merge
directly into the existing comparison tables without re-spending GPU-hours re-running
AutoETS/AutoTheta/LightGBM (those don't need a GPU and their numbers are already known-reproducible
under this exact fixed setup).

```bash
# panel.parquet must be copied over first — it's gitignored, not part of the git clone (see the
# main session's instructions for the scp command)
sbatch scripts/bridges2/run_deep_backtest.sbatch
squeue -u $USER   # wait for it to clear
cat stockup-nbeats_<jobid>.out
cat stockup-nbeats_<jobid>.err
```

Writes `reports/backtest_deep_<date>.md` and `reports/decision_deep_<date>.md` — copy both back to
the M2 (`scp bridges2:~/stockup/reports/*_deep_*.md reports/`) for the final README write-up.

## Status

Diagnostic confirmed, integration written, awaiting the real 400-series/4-fold run.
