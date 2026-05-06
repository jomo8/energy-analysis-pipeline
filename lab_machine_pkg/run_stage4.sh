#!/bin/bash
# Run Stage 4 (regression with baseline vs +NLP features). Produces
# results/regression_results.csv.
#
# Energy logger should already be running before this script starts.

set -e

source .venv/bin/activate
export HF_HOME="$(pwd)/.hf_cache"
mkdir -p "$HF_HOME"
mkdir -p logs

LOG="logs/stage4_run_$(date +%Y%m%d_%H%M%S).log"
echo "Running Stage 4. Output -> $LOG"

jupyter nbconvert --to notebook --execute 04_regression.ipynb \
    --output 04_regression.ipynb \
    --ExecutePreprocessor.timeout=3600 \
    2>&1 | tee "$LOG"

echo "Stage 4 complete. Log: $LOG"
