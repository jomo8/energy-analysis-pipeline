#!/bin/bash
# Sequential run of Stages 3, 4, 5. Energy logger should already be running.
# Exits on first failure so you don't burn through Stage 4 if Stage 3 broke.

set -e

echo "=== Starting full pipeline run at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo ""
echo "=== Stage 3 (NLP fine-tuning, ~30-60 min) ==="
./run_stage3.sh

echo ""
echo "=== Stage 4 (regression, ~5-15 min) ==="
./run_stage4.sh

echo ""
echo "=== Stage 5 (deployment inference, ~5-15 min) ==="
./run_stage5.sh

echo ""
echo "=== All stages complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Results in:"
echo "  - results/nlp_results.csv (Stage 3 metrics)"
echo "  - results/regression_results.csv (Stage 4 metrics)"
echo "  - results/inference_results.csv (Stage 5 metrics)"
echo "  - artifacts/nlp_probs.parquet (Stage 3 -> Stage 4 handoff)"
echo "  - artifacts/nlp_probs_run_meta.json (best-run metadata for energy join)"
echo ""
echo "Stop the energy logger now and save its output. Both this folder and"
echo "the energy log are needed to compute the energy/accuracy tradeoff."
