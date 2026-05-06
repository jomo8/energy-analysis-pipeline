#!/bin/bash
# Run Stage 5 (deployment inference benchmark). Produces
# results/inference_results.csv.
#
# Energy logger should already be running before this script starts.

set -e

source .venv/bin/activate
export HF_HOME="$(pwd)/.hf_cache"
mkdir -p "$HF_HOME"
mkdir -p logs

LOG="logs/stage5_run_$(date +%Y%m%d_%H%M%S).log"
echo "Running Stage 5. Output -> $LOG"

jupyter nbconvert --to notebook --execute 05_deployment_inference.ipynb \
    --output 05_deployment_inference.ipynb \
    --ExecutePreprocessor.timeout=3600 \
    2>&1 | tee "$LOG"

echo "Stage 5 complete. Log: $LOG"
