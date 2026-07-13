#!/bin/bash
# Evaluate a SIREN model trained on HH-RLHF against HarmBench and AdvBench.
#
# Usage:
#   bash eval_harm_siren.sh [MODEL]
#   MODEL: llama3-8b-sft (default) | llama3-8b-instruct
#
# HarmBench/AdvBench only contain harmful prompts. By default we evaluate on
# the harmful prompts only, so RECALL == detection rate is the metric to read
# (F1/precision are degenerate without negatives). Set HARM_EVAL_ADD_SAFE=1 to
# mix in an equal number of safe Alpaca instructions as negatives (label 0),
# which makes F1/precision meaningful.

MODEL="${1:-llama3-8b-sft}"
DEVICE="cuda"
BATCH_SIZE="${BATCH_SIZE:-8}"

export HARM_EVAL_ADD_SAFE="${HARM_EVAL_ADD_SAFE:-0}"

DATASETS=(
    "harmbench"
    "advbench"
)

echo "========================================"
echo "Evaluating SIREN"
echo "========================================"
echo "Model: $MODEL"
echo "Datasets: ${DATASETS[@]}"
echo "HARM_EVAL_ADD_SAFE: $HARM_EVAL_ADD_SAFE"
echo ""

python evaluate_general_siren.py \
    --model $MODEL \
    --datasets ${DATASETS[@]} \
    --device $DEVICE \
    --batch_size $BATCH_SIZE

echo ""
echo "Done!"
