#!/bin/bash
# Train SIREN on HH-RLHF (harmless-base) for the Llama-3-8B SFT/Instruct comparison.
#
# Usage:
#   bash run_hh_siren.sh [MODEL] [REP_TYPE]
#   MODEL:    llama3-8b-sft (default) | llama3-8b-instruct
#   REP_TYPE: residual_mean (default) | mlp_mean | mlpneuron_mean
#             mlpneuron_mean probes the down_proj input (14336-dim MLP neurons),
#             the same space as our earlier RMS change-score analysis.

MODEL="${1:-llama3-8b-sft}"
REP_TYPES="${2:-residual_mean}"

DEVICE="cuda"
BATCH_SIZE="${BATCH_SIZE:-16}"   # 8 fits a 24GB GPU; 16 needs ~40GB
# C = inverse L1 strength (l1_lambda = 1/C in probe_trainer.py). The default
# grid was tuned for 4096-dim residual_mean; the 14336-dim mlpneuron_mean
# space needs weaker regularization, e.g. C_VALUES="1000.0 5000.0 20000.0".
C_VALUES="${C_VALUES:-200.0 500.0 1000.0}"
# threshold=0.9 on mlpneuron_mean selects ~266k features and OOMs stage 3
# unless the job has ~96GB RAM; use THRESHOLDS="0.6 0.8" there.
THRESHOLDS="${THRESHOLDS:-0.6 0.8 0.9}"
N_TRIALS=32
N_JOBS=1
N_FOLDS=5
VAL_RATIO=0.2
USE_GPU_DATA=1

# Number of HH-RLHF preference pairs sampled for training (2 texts per pair).
export HH_RLHF_MAX_ROWS="${HH_RLHF_MAX_ROWS:-8000}"

# Output names get this suffix so different runs never overwrite each other:
#   probes/{MODEL}_general_probes{SUFFIX}.pkl
#   probes/optuna/{MODEL}_general{SUFFIX}/
# Defaults to "" for residual_mean (backward compatible with existing paths)
# and "-<rep_type>" otherwise. Override with OUTPUT_SUFFIX for tuning runs,
# e.g. OUTPUT_SUFFIX="-mlpneuron_mean-c20k".
if [ -z "${OUTPUT_SUFFIX+x}" ]; then
    if [ "$REP_TYPES" = "residual_mean" ]; then
        OUTPUT_SUFFIX=""
    else
        OUTPUT_SUFFIX="-${REP_TYPES// /+}"
    fi
fi

DATASETS=(
    "hh_rlhf"
)

echo "========================================"
echo "Training SIREN on HH-RLHF"
echo "========================================"
echo "Model: $MODEL"
echo "Rep type: $REP_TYPES"
echo "HH_RLHF_MAX_ROWS: $HH_RLHF_MAX_ROWS"
echo "C values: $C_VALUES"
echo "Thresholds: $THRESHOLDS"
echo "Output suffix: '$OUTPUT_SUFFIX'"
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
    --device $DEVICE \
    --output_suffix "$OUTPUT_SUFFIX"

if [ $? -ne 0 ]; then
    echo "ERROR: train_general_siren.py failed"
    exit 1
fi

echo ""
echo "Done! Probes: probes/${MODEL}_general_probes${OUTPUT_SUFFIX}.pkl"
echo "      Model:  probes/optuna/${MODEL}_general${OUTPUT_SUFFIX}/"
