#!/bin/bash
# Evaluate a SIREN model on the standardized HarmBench + Alpaca test split.
#
# Uses the fixed 100 harmful + 100 safe test rows from data_files/ (balanced),
# so F1/precision/recall are all meaningful. Reads the chat-templated
# `formatted_input` column, so extraction runs with --pre_templated 1 to match
# how the model was trained.
#
# Usage:
#   bash eval_standard_siren.sh [MODEL]
#   MODEL: llama3-8b-instruct (default) | llama3-8b-sft | ...
#
# Requires train/probes/optuna/${MODEL}_general/best_model.pkl from training.

MODEL="${1:-llama3-8b-instruct}"
DEVICE="cuda"
BATCH_SIZE="${BATCH_SIZE:-8}"
# Must match training: 1 for the chat-templated formatted_input column.
PRE_TEMPLATED="${PRE_TEMPLATED:-1}"

DATASETS=(
    "standard"
)

echo "========================================"
echo "Evaluating SIREN on standardized HarmBench + Alpaca"
echo "========================================"
echo "Model: $MODEL"
echo "Text column: ${SIREN_TEXT_COLUMN:-formatted_input}"
echo ""

python evaluate_general_siren.py \
    --model $MODEL \
    --datasets ${DATASETS[@]} \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --pre_templated $PRE_TEMPLATED

echo ""
echo "Done!"
