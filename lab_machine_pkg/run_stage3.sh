#!/bin/bash
# Run Stage 3 (NLP fine-tuning) end-to-end. Produces results/nlp_results.csv
# and artifacts/nlp_probs.parquet.
#
# Energy logger should already be running before this script starts.

set -e

source .venv/bin/activate
export HF_HOME="$(pwd)/.hf_cache"
mkdir -p "$HF_HOME"
mkdir -p logs

LOG="logs/stage3_run_$(date +%Y%m%d_%H%M%S).log"
echo "Running Stage 3. Output -> $LOG"

jupyter nbconvert --to notebook --execute 03_sentiment_nlp.ipynb \
    --output 03_sentiment_nlp.ipynb \
    --ExecutePreprocessor.timeout=10800 \
    2>&1 | tee "$LOG"

echo "Stage 3 complete. Log: $LOG"
