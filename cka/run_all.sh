#!/bin/bash
# Full CKA pipeline: build held-out prompt sets, extract activations, run the
# cross-method and cross-layer analyses.
#
# Run from the repo root:
#   bash cka/run_all.sh                                    # the default start
#   BUDGETS="2500 5000 10000" bash cka/run_all.sh          # budget sweep
#   METHODS=all BUDGETS="2500 5000 10000" bash cka/run_all.sh   # everything
#   POOLINGS="mean last" bash cka/run_all.sh               # + robustness pooling
#
# Defaults are deliberately the smallest run that answers the question: four
# canonical methods at one budget. Widen once you have seen those numbers --
# METHODS=all is 28 pairs instead of 6.
#
# Steps 1-2 need a GPU and run ONCE per dataset regardless of how many budgets
# or methods you sweep. Steps 3-4 are CPU-only and reread the saved
# activations, so sweeping budgets is cheap in GPU terms.

set -e

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
# Each entry is one prompt set, analysed separately. Use "+" inside an entry to
# POOL corpora into a single set, which is how you get past a dataset's
# minority-class limit -- WildGuard's test split balances out at only 1508
# prompts because just 754 of its 1725 rows are harmful.
#   "wildguard xstest"                  two separate sets (default)
#   "wildguard+openai_moderation"       one pooled set, ~2000+ prompts
# Swap in "openai_moderation beavertails" if your HF account lacks access.
DATASETS="${DATASETS:-wildguard xstest}"
POOLINGS="${POOLINGS:-mean}"
# Accepts several: "2500 5000 10000". BUDGET (singular) still works.
BUDGETS="${BUDGETS:-${BUDGET:-2500}}"
MAX_PROMPTS="${MAX_PROMPTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# "all" expands to all 8 selections: siren, wang x2, zhao x2, yang x3.
METHODS="${METHODS:-siren wang zhao_topk yang_rms}"
NULL_SEEDS="${NULL_SEEDS:-20}"
CROSS_LAYER_SEEDS="${CROSS_LAYER_SEEDS:-5}"
# The cross-layer sweep is the slow step; at METHODS=all it is 28 heatmaps.
RUN_CROSS_LAYER="${RUN_CROSS_LAYER:-1}"
CROSS_LAYER_METHODS="${CROSS_LAYER_METHODS:-$METHODS}"
CROSS_LAYER_BUDGET="${CROSS_LAYER_BUDGET:-2500}"
# Suffix appended to every analysis output filename. Set it whenever you run a
# second configuration against the SAME dataset, or the two runs overwrite each
# other's CSVs and figures:
#   RUN_TAG=all8 METHODS=all bash cka/run_all.sh
RUN_TAG="${RUN_TAG:-}"
# 1 = require the activations to already exist and fail if they do not, instead
# of extracting. Use this for a second job that reuses a first job's
# activations: it needs no GPU at all, since steps 3-4 are CPU-only.
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================================="
echo "model      : $MODEL_PATH"
echo "datasets   : $DATASETS"
echo "poolings   : $POOLINGS"
echo "budgets    : $BUDGETS"
echo "methods    : $METHODS"
echo "cross-layer: $RUN_CROSS_LAYER (N=$CROSS_LAYER_BUDGET, $CROSS_LAYER_METHODS)"
echo "run tag    : ${RUN_TAG:-<none>}   skip extract: $SKIP_EXTRACT"
echo "=============================================================="

for DATASET in $DATASETS; do
    # "a+b" pools corpora a and b into one prompt set tagged "a+b".
    MEMBERS="${DATASET//+/ }"

    echo ""
    echo "### [1/4] Building held-out prompt set: $DATASET"
    if [ -f "cka/prompts/${DATASET}.csv" ]; then
        echo "  cka/prompts/${DATASET}.csv exists, skipping"
    else
        python cka/build_prompts.py \
            --dataset $MEMBERS \
            --max_prompts "$MAX_PROMPTS" \
            --model_path "$MODEL_PATH"
    fi

    for POOLING in $POOLINGS; do
        ACTS="cka/activations/${DATASET}_${POOLING}.npy"
        # Analysis outputs are labelled with the run tag so a second
        # configuration on the same dataset cannot overwrite the first.
        LABEL="${DATASET}_${POOLING}${RUN_TAG:+_$RUN_TAG}"

        echo ""
        echo "### [2/4] Extracting MLP-neuron activations: $DATASET / $POOLING"
        if [ -f "$ACTS" ]; then
            echo "  $ACTS exists, skipping"
        elif [ "$SKIP_EXTRACT" = "1" ]; then
            echo "  ERROR: SKIP_EXTRACT=1 but $ACTS does not exist."
            echo "  Another job may still be extracting it (the file is only"
            echo "  renamed into place once complete). Wait for it, or unset"
            echo "  SKIP_EXTRACT to extract here."
            exit 1
        else
            python cka/extract_activations.py \
                --prompts "cka/prompts/${DATASET}.csv" \
                --model_path "$MODEL_PATH" \
                --pooling "$POOLING" \
                --batch_size "$BATCH_SIZE"
        fi

        for BUDGET in $BUDGETS; do
            echo ""
            echo "### [3/4] Cross-method CKA: $DATASET / $POOLING / N=$BUDGET"
            python cka/run_cka.py \
                --activations "$ACTS" \
                --methods $METHODS \
                --budget "$BUDGET" \
                --null_seeds "$NULL_SEEDS" \
                --label "$LABEL"
        done

        if [ "$RUN_CROSS_LAYER" = "1" ]; then
            echo ""
            echo "### [4/4] Cross-layer CKA: $DATASET / $POOLING / N=$CROSS_LAYER_BUDGET"
            python cka/run_cross_layer.py \
                --activations "$ACTS" \
                --methods $CROSS_LAYER_METHODS \
                --budget "$CROSS_LAYER_BUDGET" \
                --variant "class+length" \
                --null_seeds "$CROSS_LAYER_SEEDS" \
                --label "$LABEL"
        fi
    done
done

echo ""
echo "Done. Results in cka/results/"
ls -1 cka/results/ | head -40
