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
assumed — `scripts/bridges2/diagnose_nbeats.py` (submitted via `diagnose.sbatch`) trains a tiny
NBEATS on synthetic data and prints the real signature and output shape for the version actually
installed, the same empirical-first approach used for the statsforecast wrapper in Phase 2.

```bash
sbatch scripts/bridges2/diagnose.sbatch
# wait for it to finish (squeue -u $USER), then:
cat stockup-diag_<jobid>.out
```

`reorderpoint/models/deep.py` and the real training/backtest job get written from that output,
not before.

## Status

Diagnostic in progress — this file gets the real `NBEATS` integration details once
`diagnose_nbeats.py`'s output is in hand.
