#!/bin/bash
# Run Stage 3 (NLP fine-tuning) for KO.
# Energy logger should already be running before this script starts.
set -e
source .venv-ko/bin/activate
mkdir -p logs
LOG="logs/stage3_run_$(date +%Y%m%d_%H%M%S).log"
echo "Running Stage 3 (KO). Output -> $LOG"
jupyter nbconvert --to notebook --execute 03_sentiment_nlp.ipynb \
    --output 03_sentiment_nlp.ipynb \
    --ExecutePreprocessor.timeout=14400 \
    2>&1 | tee "$LOG"
echo "Stage 3 complete. Log: $LOG"
