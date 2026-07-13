#!/bin/bash
# Train SIREN on a SINGLE dataset from the paper (clean harmfulness labels),
# as an alternative to the weak-labeled HH-RLHF run.
#
# Usage:
#   bash run_paper_siren.sh [MODEL] [DATASET] [REP_TYPE]
#   MODEL:   llama3-8b-instruct (default) | llama3-8b-sft | ...
#   DATASET: aegis2 (default) | wildguard | aegis | toxic_chat |
#            openai_moderation | safe_rlhf | beavertails
#   REP_TYPE: residual_mean (default) | mlp_mean | mlpneuron_mean
#
# Aegis 2.0 is the default: it labels PROMPT harmfulness (matching the
# HarmBench/AdvBench evaluation, which are harmful-prompt sets), is used by the
# paper, and is PUBLIC (WildGuard is equally suitable but gated on HF).
# Response-level sets (safe_rlhf, beavertails) label prompt+response instead.
#
# NOTE: the trained model is saved to probes/optuna/${MODEL}_general/ -- the
# SAME path as run_hh_siren.sh. Move/rename that folder first if you want to
# keep the HH-RLHF run's results (the sbatch wrapper archives it automatically).

MODEL="${1:-llama3-8b-instruct}"
DATASET="${2:-aegis2}"
REP_TYPES="${3:-residual_mean}"

DEVICE="cuda"
BATCH_SIZE="${BATCH_SIZE:-16}"   # 8 fits a 24GB GPU; 16 needs ~40GB
C_VALUES="200.0 500.0 1000.0"
THRESHOLDS="0.6 0.8 0.9"
N_TRIALS=32
N_JOBS=1
N_FOLDS=5
VAL_RATIO=0.2
USE_GPU_DATA=1

echo "========================================"
echo "Training SIREN on paper dataset"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Rep type: $REP_TYPES"
echo ""

python train_general_siren.py \
    --model $MODEL \
    --datasets $DATASET \
    --batch_size $BATCH_SIZE \
    --c_values $C_VALUES \
    --pooling_types $REP_TYPES \
    --thresholds $THRESHOLDS \
    --n_trials $N_TRIALS \
    --n_jobs $N_JOBS \
    --n_folds $N_FOLDS \
    --val_ratio $VAL_RATIO \
    --use_gpu_data $USE_GPU_DATA \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "ERROR: train_general_siren.py failed"
    exit 1
fi

echo ""
echo "Done! Model saved to probes/optuna/${MODEL}_general/"
