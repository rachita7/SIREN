#!/bin/bash
# Evaluate a SIREN model trained on HH-RLHF against HarmBench and AdvBench.
#
# Usage:
#   bash eval_harm_siren.sh [MODEL]
#   MODEL: llama3-8b-sft (default) | llama3-8b-instruct
#
# HarmBench/AdvBench only contain harmful prompts; by default an equal number
# of safe Alpaca instructions is mixed in as negatives (label 0) so that
# F1/precision are meaningful. Set HARM_EVAL_ADD_SAFE=0 to evaluate on the
# harmful prompts only (then recall == detection rate is the metric to read).

MODEL="${1:-llama3-8b-sft}"
DEVICE="cuda"
BATCH_SIZE=16

export HARM_EVAL_ADD_SAFE="${HARM_EVAL_ADD_SAFE:-1}"

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
