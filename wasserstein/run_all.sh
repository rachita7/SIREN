#!/bin/bash
# Wasserstein comparison of the safety-neuron selections, with the same
# controls as cka/run_all.sh, followed by the CKA-vs-Wasserstein agreement
# report.
#
# Run from the repo root:
#   bash wasserstein/run_all.sh                                   # default
#   METHODS=all BUDGETS="459 2294 4588 9175" bash wasserstein/run_all.sh
#   SKIP_EXTRACT=1 bash wasserstein/run_all.sh                    # CPU only
#
# The activation tensors are the ones cka/ extracts (cka/activations/); if they
# exist nothing here needs a GPU. If they do not and SKIP_EXTRACT=0, the cka/
# prompt-building and extraction steps are run first (GPU, once per dataset).
#
# The comparison step needs cka/results/cka_{tag}_N{budget}.csv for the SAME
# dataset, pooling, budget and RUN_TAG. If that file is missing the comparison
# falls back to the CKA columns run_wasserstein.py computes internally on the
# identical inputs (still a valid answer to "do they agree?"); run
# cka/run_all.sh with matching settings to also compare against the reference
# CKA pipeline.

set -e

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASETS="${DATASETS:-wildguard xstest}"
POOLINGS="${POOLINGS:-mean}"
BUDGETS="${BUDGETS:-${BUDGET:-2294}}"
MAX_PROMPTS="${MAX_PROMPTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
METHODS="${METHODS:-siren wang zhao_topk yang_refusal}"
NULL_SEEDS="${NULL_SEEDS:-20}"
CEILING_SEEDS="${CEILING_SEEDS:-10}"
RUN_TAG="${RUN_TAG:-}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
# Layer W1 does not use activations and writes wd_layer_selections_*.csv.
# A second dataset job must skip it or it overwrites the first job's files.
SKIP_LAYER_W1="${SKIP_LAYER_W1:-0}"
# Extra flags for run_wasserstein.py, e.g. WD_FLAGS="--signed --save_matching"
WD_FLAGS="${WD_FLAGS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================================="
echo "datasets   : $DATASETS"
echo "poolings   : $POOLINGS"
echo "budgets    : $BUDGETS"
echo "methods    : $METHODS"
echo "run tag    : ${RUN_TAG:-<none>}   skip extract: $SKIP_EXTRACT"
echo "skip layer : $SKIP_LAYER_W1"
echo "wd flags   : ${WD_FLAGS:-<none>}"
echo "=============================================================="

if [ "$SKIP_LAYER_W1" = "1" ]; then
    echo ""
    echo "### [0/3] Layer W1 skipped (SKIP_LAYER_W1=1; files are dataset-independent)"
else
    echo ""
    echo "### [0/3] Layer W1 (needs no activations)"
    for BUDGET in $BUDGETS; do
        python wasserstein/run_wasserstein.py --layer_only \
            --methods $METHODS --budget "$BUDGET" \
            --label "selections${RUN_TAG:+_$RUN_TAG}"
    done
fi

for DATASET in $DATASETS; do
    MEMBERS="${DATASET//+/ }"
    for POOLING in $POOLINGS; do
        ACTS="cka/activations/${DATASET}_${POOLING}.npy"
        LABEL="${DATASET}_${POOLING}${RUN_TAG:+_$RUN_TAG}"

        echo ""
        echo "### [1/3] Activations: $DATASET / $POOLING"
        if [ -f "$ACTS" ]; then
            echo "  $ACTS exists, reusing"
        elif [ "$SKIP_EXTRACT" = "1" ]; then
            echo "  ERROR: SKIP_EXTRACT=1 but $ACTS does not exist."
            echo "  Run cka/run_all.sh (GPU) first, or unset SKIP_EXTRACT."
            exit 1
        else
            if [ ! -f "cka/prompts/${DATASET}.csv" ]; then
                python cka/build_prompts.py --dataset $MEMBERS \
                    --max_prompts "$MAX_PROMPTS" --model_path "$MODEL_PATH"
            fi
            python cka/extract_activations.py \
                --prompts "cka/prompts/${DATASET}.csv" \
                --model_path "$MODEL_PATH" \
                --pooling "$POOLING" --batch_size "$BATCH_SIZE"
        fi

        for BUDGET in $BUDGETS; do
            echo ""
            echo "### [2/3] Wasserstein: $DATASET / $POOLING / N=$BUDGET"
            python wasserstein/run_wasserstein.py \
                --activations "$ACTS" \
                --methods $METHODS \
                --budget "$BUDGET" \
                --null_seeds "$NULL_SEEDS" \
                --ceiling_seeds "$CEILING_SEEDS" \
                --label "$LABEL" $WD_FLAGS

            echo ""
            echo "### [3/3] Agreement with CKA: $DATASET / $POOLING / N=$BUDGET"
            WD_CSV="wasserstein/results/wd_${LABEL}_N${BUDGET}.csv"
            CKA_CSV="cka/results/cka_${LABEL}_N${BUDGET}.csv"
            if [ -f "$CKA_CSV" ]; then
                python wasserstein/compare_to_cka.py \
                    --wasserstein "$WD_CSV" --cka "$CKA_CSV"
            else
                echo "  $CKA_CSV not found; comparing against the CKA computed"
                echo "  inside the Wasserstein run (same inputs and seeds)."
                python wasserstein/compare_to_cka.py --wasserstein "$WD_CSV"
            fi
        done
    done
done

echo ""
echo "Done. Results in wasserstein/results/"
ls -1 wasserstein/results/ | head -40
