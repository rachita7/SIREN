#!/bin/bash
# Train SIREN on the standardized HarmBench (harmful) + Alpaca (safe) benchmark.
#
# Reads the fixed, pre-split, chat-templated CSVs from data_files/
# (260 train / 40 val / 100 test per corpus, class-balanced per split). Because
# the CSVs' `formatted_input` column already carries the Llama-3 chat template
# with literal special tokens, extraction runs with --pre_templated 1
# (add_special_tokens=False) so the tokenizer does not add a second BOS.
#
# Usage:
#   bash run_standard_siren.sh [MODEL] [REP_TYPE]
#   MODEL:    llama3-8b-instruct (default) | llama3-8b-sft | ...
#   REP_TYPE: residual_mean (default) | mlp_mean | mlpneuron_mean

MODEL="${1:-llama3-8b-instruct}"
REP_TYPES="${2:-residual_mean}"

DEVICE="cuda"
BATCH_SIZE="${BATCH_SIZE:-16}"   # 8 fits a 24GB GPU; 16 needs ~40GB
C_VALUES="200.0 500.0 1000.0"
THRESHOLDS="0.6 0.8 0.9"
N_TRIALS=32
N_JOBS=1
N_FOLDS=5
VAL_RATIO=0.2                    # ignored by the 'standard' loader (fixed splits)
USE_GPU_DATA=1

# 1 = texts are chat-templated (formatted_input) -> add_special_tokens=False.
# To use the raw prompt column instead, run with:
#   SIREN_TEXT_COLUMN=prompt PRE_TEMPLATED=0 bash run_standard_siren.sh ...
PRE_TEMPLATED="${PRE_TEMPLATED:-1}"

DATASETS=(
    "standard"
)

echo "========================================"
echo "Training SIREN on standardized HarmBench + Alpaca"
echo "========================================"
echo "Model: $MODEL"
echo "Rep type: $REP_TYPES"
echo "Text column: ${SIREN_TEXT_COLUMN:-formatted_input}"
echo ""

python train_general_siren.py \
    --model $MODEL \
    --datasets ${DATASETS[@]} \
    --batch_size $BATCH_SIZE \
    --c_values $C_VALUES \
    --pooling_types $REP_TYPES \
    --thresholds $THRESHOLDS \
    --n_trials $N_TRIALS \
    --n_jobs $N_JOBS \
    --n_folds $N_FOLDS \
    --val_ratio $VAL_RATIO \
    --use_gpu_data $USE_GPU_DATA \
    --pre_templated $PRE_TEMPLATED \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "ERROR: train_general_siren.py failed"
    exit 1
fi

echo ""
echo "Done! Model saved to probes/optuna/${MODEL}_general/"
