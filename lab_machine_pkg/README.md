# Energy Analysis Pipeline — Lab Machine Package

This package runs the energy-tracked stages (3, 4, 5) of an accuracy/energy tradeoff study on financial NLP and regression models. Stages 1, 2, 6, and 7 are run elsewhere; this folder is for the lab-machine measurement runs only.

## Prerequisites on the lab machine

- **OS:** Ubuntu 20.04 or newer (tested on 20.04).
- **Python:** 3.10 (NOT the default 3.8 on Ubuntu 20). Install if missing:
```bash
  sudo apt install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.10 python3.10-venv python3.10-dev
```
- **GPU:** any NVIDIA GPU with a working driver. Verify with `nvidia-smi`.
- **Disk:** at least 5 GB free for the venv, HuggingFace model cache, and run logs.
- **Network:** internet access for pip install and HuggingFace model downloads.

## Setup (do this once)

```bash
cd lab_machine_pkg
./setup.sh
```

This creates a `.venv/` directory and installs all pinned dependencies. Takes 5-15 minutes (torch is the slow one).

After setup, verify the GPU is visible to PyTorch:

```bash
source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

You should see `CUDA: True` and a GPU name. If it says `False`, the lab GPU isn't being detected by torch — fix this before running stages.

## Running the pipeline

**Before you start:** make sure the energy logger is already running. The notebooks record ISO 8601 UTC timestamps for each train/inference operation; these will be joined to the energy log post-run.

To run all three stages sequentially:

```bash
./run_all.sh
```

Approximate runtimes on a modern GPU:
- Stage 3 (4 NLP models × 10 seeds = 40 fine-tuning runs): 30-60 minutes.
- Stage 4 (7 regressors × 10 seeds × 2 variants = 140 fits): 5-15 minutes.
- Stage 5 (4 NLP models × 1000-sample inference benchmark): 5-15 minutes.

Total: roughly 45-90 minutes. The machine is fully GPU-bound during Stage 3.

To run individual stages (for debugging or partial re-runs):

```bash
./run_stage3.sh
./run_stage4.sh
./run_stage5.sh
```

## What gets produced

After a successful run, you'll have:

- `results/nlp_results.csv` — one row per (model, seed) Stage 3 run. Columns include `accuracy`, `f1`, `train_time_s`, `infer_time_s`, and ISO timestamps for the energy join.
- `results/regression_results.csv` — one row per (regressor, seed, variant) Stage 4 run. Columns include `r2`, `mse`, `mae`, timings, and ISO timestamps.
- `results/inference_results.csv` — one row per NLP model in Stage 5. Columns include `time_per_query`, `time_per_token`, and ISO timestamps for the inference window.
- `artifacts/nlp_probs.parquet` — per-headline 3-class probabilities from the best Stage 3 run. (Used by Stage 4's `+NLP` variant; included here for completeness.)
- `artifacts/nlp_probs_run_meta.json` — sidecar metadata (model, seed, timestamps) for the best-model retrain in Stage 3.
- `logs/stage*_run_*.log` — per-stage execution logs.

The `wall_*_iso` columns in the result CSVs are the join keys for the external energy log. A separate join script (not included here, written after the lab visit when the energy log format is known) will integrate power over each `[start, end]` window to produce `energy_kwh` columns.

## Troubleshooting

**`python3.10: command not found`** — install Python 3.10 (see Prerequisites).

**`torch.cuda.is_available() returns False`** — the GPU driver may be misconfigured, or the wrong torch wheel got installed. Run `nvidia-smi` to confirm the driver works. If the driver is fine but torch can't see the GPU, you may need to reinstall torch with the right CUDA version: `pip uninstall torch && pip install torch` (modern pip auto-detects CUDA).

**`'name' is a required property` from nbconvert** — a notebook has malformed output cells. Strip outputs:
```bash
python -c "
import nbformat
for path in ['03_sentiment_nlp.ipynb', '04_regression.ipynb', '05_deployment_inference.ipynb']:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(nb, path)
"
```

**Stage 3 OOMs** — reduce `train_batch_size` in `common.py` (under `CONFIG["nlp"]`). Default is 16; try 8.

**HuggingFace downloads fail** — confirm internet access with `curl -I https://huggingface.co`. If blocked, you may need a proxy or to pre-download models.

## What's NOT in this package

- **Energy join script.** Will be written after the lab visit, once the format of the external energy logger's output is known.
- **Stage 1 (data ingestion) and Stage 2 (preprocessing).** Already run on the cluster. The artifacts they produced are bundled here in `artifacts/`.
- **Stage 6 (scoring/ranking) and Stage 7 (visualization).** Run elsewhere after the energy join, on the result CSVs.
- **Git history.** This folder is a deliverable, not a working repository.
