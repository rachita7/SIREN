#!/bin/bash
# Full CKA pipeline: build held-out prompt sets, extract activations, run the
# cross-method and cross-layer analyses.
#
# Run from the repo root:
#   bash cka/run_all.sh
#   DATASETS="wildguard xstest" BUDGET=2500 bash cka/run_all.sh
#   POOLINGS="mean last" bash cka/run_all.sh          # + the robustness pooling
#
# Steps 1-2 need a GPU (~15 min per dataset on a 24GB card for 2000 prompts).
# Steps 3-4 are CPU-only and can be rerun freely on the saved activations.

set -e

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
# wildguard = primary held-out set; xstest = the confound stress test.
# Swap in "openai_moderation beavertails" if your HF account lacks access.
DATASETS="${DATASETS:-wildguard xstest}"
POOLINGS="${POOLINGS:-mean}"
BUDGET="${BUDGET:-2500}"
MAX_PROMPTS="${MAX_PROMPTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
METHODS="${METHODS:-siren wang zhao_topk yang_rms}"
NULL_SEEDS="${NULL_SEEDS:-20}"
CROSS_LAYER_SEEDS="${CROSS_LAYER_SEEDS:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================================="
echo "model      : $MODEL_PATH"
echo "datasets   : $DATASETS"
echo "poolings   : $POOLINGS"
echo "budget     : N=$BUDGET   methods: $METHODS"
echo "=============================================================="

for DATASET in $DATASETS; do
    echo ""
    echo "### [1/4] Building held-out prompt set: $DATASET"
    if [ -f "cka/prompts/${DATASET}.csv" ]; then
        echo "  cka/prompts/${DATASET}.csv exists, skipping"
    else
        python cka/build_prompts.py \
            --dataset "$DATASET" \
            --max_prompts "$MAX_PROMPTS" \
            --model_path "$MODEL_PATH"
    fi

    for POOLING in $POOLINGS; do
        echo ""
        echo "### [2/4] Extracting MLP-neuron activations: $DATASET / $POOLING"
        if [ -f "cka/activations/${DATASET}_${POOLING}.npy" ]; then
            echo "  cka/activations/${DATASET}_${POOLING}.npy exists, skipping"
        else
            python cka/extract_activations.py \
                --prompts "cka/prompts/${DATASET}.csv" \
                --model_path "$MODEL_PATH" \
                --pooling "$POOLING" \
                --batch_size "$BATCH_SIZE"
        fi

        echo ""
        echo "### [3/4] Cross-method CKA: $DATASET / $POOLING"
        python cka/run_cka.py \
            --activations "cka/activations/${DATASET}_${POOLING}.npy" \
            --methods $METHODS \
            --budget "$BUDGET" \
            --null_seeds "$NULL_SEEDS"

        echo ""
        echo "### [4/4] Cross-layer CKA: $DATASET / $POOLING"
        python cka/run_cross_layer.py \
            --activations "cka/activations/${DATASET}_${POOLING}.npy" \
            --methods $METHODS \
            --budget "$BUDGET" \
            --variant "class+length" \
            --null_seeds "$CROSS_LAYER_SEEDS"
    done
done

echo ""
echo "Done. Results in cka/results/"
ls -1 cka/results/ | head -40
