#!/bin/bash
# SIREN stability protocol: three independent probe trainings on disjoint
# thirds of the standardized HarmBench+Alpaca data, then a LAYER-LEVEL
# stability comparison with analysis/siren_layer_stability.py.
#
# Because SIREN's selection is created by probe training, stability must be
# tested by re-training on resampled data. Default resampling is thirds of
# the TRAIN set (~173 texts per run; val/test keep their normal roles):
#   train = third i of harmbench_train+alpaca_train
#   val   = standard val files (C selection / early stopping)
#   test  = standard test files (reporting only)
#
# Usage:
#   bash run_stability_siren.sh [MODEL] [REP_TYPE]
#   MODEL:    llama3-8b-instruct (default) | llama3-8b-sft | ...
#   REP_TYPE: residual_mean (default) | mlpneuron_mean
#
# SPLIT_DATASET=standard_split uses the three TEST-set thirds as training
# data instead (matches the Wang/Zhao reproduction's exact split corpora,
# but only ~66 texts per run and the test set is consumed by training).
#
# Afterwards, from the repo root:
#   python analysis/siren_layer_stability.py --model $MODEL --pooling_type $REP_TYPE

MODEL="${1:-llama3-8b-instruct}"
REP_TYPES="${2:-residual_mean}"

DEVICE="cuda"
BATCH_SIZE="${BATCH_SIZE:-8}"
C_VALUES="${C_VALUES:-200.0 500.0 1000.0}"
PRE_TEMPLATED="${PRE_TEMPLATED:-1}"   # formatted_input column is chat-templated
SPLIT_DATASET="${SPLIT_DATASET:-standard_trainsplit}"
SPLIT_TAG="${SPLIT_DATASET#standard_}"

echo "========================================"
echo "SIREN stability runs (3 x $SPLIT_DATASET)"
echo "========================================"
echo "Model: $MODEL"
echo "Rep type: $REP_TYPES"
echo "C values: $C_VALUES"
echo ""

for i in 1 2 3; do
    echo ""
    echo "--- Split $i / 3 ---"
    python train_general_siren.py \
        --model $MODEL \
        --datasets ${SPLIT_DATASET}$i \
        --batch_size $BATCH_SIZE \
        --c_values $C_VALUES \
        --pooling_types $REP_TYPES \
        --pre_templated $PRE_TEMPLATED \
        --use_gpu_data 1 \
        --device $DEVICE \
        --skip_final 1 \
        --output_suffix="-stability-${REP_TYPES}-${SPLIT_TAG}${i}"
    if [ $? -ne 0 ]; then
        echo "ERROR: split $i failed"
        exit 1
    fi
done

echo ""
echo "Done! Probes:"
for i in 1 2 3; do
    echo "  probes/${MODEL}_general_probes-stability-${REP_TYPES}-${SPLIT_TAG}${i}.pkl"
done
echo "Next: python analysis/siren_layer_stability.py --model $MODEL --pooling_type $REP_TYPES"
