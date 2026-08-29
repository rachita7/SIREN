#!/bin/bash
# SIREN clean-stability protocol: three independent probe trainings on the
# pre-made thirds of the CLEANED train set (data_files/{harmbench,alpaca}_
# train_split{1,2,3}.csv), then an intersection analysis at fixed neuron
# budgets with analysis/clean_stability_intersection.py.
#
#   train = harmbench_train_split{i} + alpaca_train_split{i} (~173 texts)
#   val   = standard val files (C selection / early stopping)
#   test  = standard test files (reporting only; per-layer test F1 is stored
#           in each probes pkl)
#
# Usage:
#   bash run_clean_stability_siren.sh [MODEL] [REP_TYPE]
#   MODEL:    llama3-8b-instruct (default) | llama3-8b-sft | ...
#   REP_TYPE: mlpneuron_mean (default) | residual_mean
#
# Afterwards, from the repo root:
#   python analysis/clean_stability_intersection.py --model $MODEL \
#       --pooling_type $REP_TYPE --targets 459 2294 4588 9175

MODEL="${1:-llama3-8b-instruct}"
REP_TYPES="${2:-mlpneuron_mean}"

DEVICE="cuda"
BATCH_SIZE="${BATCH_SIZE:-8}"
# Default C grid suits the 14336-dim mlpneuron_mean space; for residual_mean
# override with C_VALUES="200.0 500.0 1000.0".
C_VALUES="${C_VALUES:-1000.0 5000.0 20000.0}"
PRE_TEMPLATED="${PRE_TEMPLATED:-1}"   # formatted_input column is chat-templated

echo "========================================"
echo "SIREN clean-stability runs (3 x standard_cleansplit)"
echo "========================================"
echo "Model: $MODEL"
echo "Rep type: $REP_TYPES"
echo "C values: $C_VALUES"
echo ""

for i in 1 2 3; do
    echo ""
    echo "--- Clean split $i / 3 ---"
    python train_general_siren.py \
        --model $MODEL \
        --datasets standard_cleansplit$i \
        --batch_size $BATCH_SIZE \
        --c_values $C_VALUES \
        --pooling_types $REP_TYPES \
        --pre_templated $PRE_TEMPLATED \
        --use_gpu_data 1 \
        --device $DEVICE \
        --skip_final 1 \
        --output_suffix="-clean_stability-${REP_TYPES}-split${i}"
    if [ $? -ne 0 ]; then
        echo "ERROR: clean split $i failed"
        exit 1
    fi
done

echo ""
echo "Done! Probes:"
for i in 1 2 3; do
    echo "  probes/${MODEL}_general_probes-clean_stability-${REP_TYPES}-split${i}.pkl"
done
echo "Next: python analysis/clean_stability_intersection.py --model $MODEL --pooling_type $REP_TYPES"
