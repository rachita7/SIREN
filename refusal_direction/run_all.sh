#!/bin/bash
# Full refusal-direction / DFA pipeline: fit the direction on its own held-out
# corpus, score every method's neurons against it, and (optionally) run the
# causal neuron-ablation test.
#
# Run from the repo root:
#   bash refusal_direction/run_all.sh                     # the default start
#   BUDGETS="459 2294 4588 9175" bash refusal_direction/run_all.sh
#   RUN_ABLATION=0 bash refusal_direction/run_all.sh      # skip the slow GPU step
#   DIRECTION_MODE=per_layer bash refusal_direction/run_all.sh  # robustness
#
# GPU steps: residual extraction for the direction (once, small), activation
# extraction per eval dataset (once, shared with cka/), and the ablation.
# run_dfa.py itself is CPU-only and rereads saved artifacts, so budget sweeps
# are cheap.
#
# One-dataset design (our controlled adaptation of Arditi et al.):
#   WildGuardTrain (fit subset) -> construct r | WildGuardTrain (val subset)
#   -> choose r's layer | WildGuardTest = cka/prompts/wildguard.csv -> the
#   ONLY prompts anything is reported on (CKA, DFA, ablation all share it).
# Neuron selection happened upstream on HarmBench+Alpaca and is not part of
# this experiment. Exact-text overlap with the eval set is dropped when the
# direction prompts are built, as a guard on top of the official split.

set -e

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
# Corpora the direction is fitted on. Default: the official TRAIN split of
# the evaluation dataset. Anything else must stay disjoint from EVAL_DATASETS.
DIRECTION_DATASETS="${DIRECTION_DATASETS:-wildguard_train}"
DIRECTION_PROMPTS="${DIRECTION_PROMPTS:-1024}"
# Eval sets for the DFA contrast and the ablation; each entry is analysed
# separately, "+" pools corpora (same convention as cka/run_all.sh).
# 'wildguard' is the official WildGuardTest split -- the frozen master
# evaluation set shared with the CKA analysis.
EVAL_DATASETS="${EVAL_DATASETS:-wildguard}"
# 'last' matches the direction's token position and is the primary setting;
# 'mean' reuses the CKA analysis' activation files if you already have them.
POOLING="${POOLING:-last}"
BUDGETS="${BUDGETS:-${BUDGET:-2294}}"
METHODS="${METHODS:-siren wang zhao_topk yang_refusal}"
NULL_SEEDS="${NULL_SEEDS:-20}"
DIRECTION_MODE="${DIRECTION_MODE:-single}"
MAX_PROMPTS="${MAX_PROMPTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# The ablation is the expensive step: each condition is a full generation run.
RUN_ABLATION="${RUN_ABLATION:-1}"
ABLATION_BUDGET="${ABLATION_BUDGET:-2294}"
ABLATION_NULL_SEEDS="${ABLATION_NULL_SEEDS:-2}"
N_HARMFUL="${N_HARMFUL:-100}"
N_BENIGN="${N_BENIGN:-100}"
# Suffix for analysis outputs; set it when running a second configuration
# against the same dataset, or the runs overwrite each other.
RUN_TAG="${RUN_TAG:-}"
# 1 = require activations/residuals to exist instead of extracting (no GPU).
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DTAG="${DIRECTION_DATASETS// /+}"
DIRECTION_NPZ="refusal_direction/directions/${DTAG}_last.npz"

echo "=============================================================="
echo "model            : $MODEL_PATH"
echo "direction corpus : $DIRECTION_DATASETS  (tag $DTAG)"
echo "eval datasets    : $EVAL_DATASETS  pooling=$POOLING"
echo "budgets          : $BUDGETS   methods: $METHODS"
echo "direction mode   : $DIRECTION_MODE"
echo "ablation         : $RUN_ABLATION (N=$ABLATION_BUDGET, ${N_HARMFUL}+${N_BENIGN} prompts)"
echo "run tag          : ${RUN_TAG:-<none>}   skip extract: $SKIP_EXTRACT"
echo "=============================================================="

echo ""
echo "### [1/5] Direction prompt sets (fit + val, disjoint from eval)"
if [ -f "refusal_direction/prompts/${DTAG}_fit.csv" ]; then
    echo "  refusal_direction/prompts/${DTAG}_fit.csv exists, skipping"
else
    python refusal_direction/build_direction_prompts.py \
        --dataset $DIRECTION_DATASETS \
        --max_prompts "$DIRECTION_PROMPTS" \
        --model_path "$MODEL_PATH"
fi

echo ""
echo "### [2/5] Residual-stream extraction for the direction (GPU, small)"
for SPLIT in fit val; do
    RESID="refusal_direction/residuals/${DTAG}_${SPLIT}_last.npy"
    if [ -f "$RESID" ]; then
        echo "  $RESID exists, skipping"
    elif [ "$SKIP_EXTRACT" = "1" ]; then
        echo "  ERROR: SKIP_EXTRACT=1 but $RESID does not exist."
        exit 1
    else
        python refusal_direction/extract_residuals.py \
            --prompts "refusal_direction/prompts/${DTAG}_${SPLIT}.csv" \
            --model_path "$MODEL_PATH" \
            --position last \
            --batch_size "$BATCH_SIZE"
    fi
done

echo ""
echo "### [3/5] Fitting the refusal direction (CPU)"
python refusal_direction/fit_direction.py \
    --fit_residuals "refusal_direction/residuals/${DTAG}_fit_last.npy" \
    --val_residuals "refusal_direction/residuals/${DTAG}_val_last.npy" \
    --model_path "$MODEL_PATH"

for DATASET in $EVAL_DATASETS; do
    MEMBERS="${DATASET//+/ }"
    ACTS="cka/activations/${DATASET}_${POOLING}.npy"
    LABEL="${DATASET}_${POOLING}_dir-${DTAG}${RUN_TAG:+_$RUN_TAG}"

    echo ""
    echo "### [4/5] Eval activations for $DATASET (shared with cka/)"
    if [ ! -f "cka/prompts/${DATASET}.csv" ]; then
        python cka/build_prompts.py \
            --dataset $MEMBERS \
            --max_prompts "$MAX_PROMPTS" \
            --model_path "$MODEL_PATH"
    fi
    if [ -f "$ACTS" ]; then
        echo "  $ACTS exists, skipping"
    elif [ "$SKIP_EXTRACT" = "1" ]; then
        echo "  ERROR: SKIP_EXTRACT=1 but $ACTS does not exist."
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
        echo "### [5/5] DFA analysis: $DATASET / N=$BUDGET (CPU)"
        python refusal_direction/run_dfa.py \
            --activations "$ACTS" \
            --direction "$DIRECTION_NPZ" \
            --methods $METHODS \
            --budget "$BUDGET" \
            --direction_mode "$DIRECTION_MODE" \
            --null_seeds "$NULL_SEEDS" \
            --label "$LABEL"
    done

    if [ "$RUN_ABLATION" = "1" ]; then
        echo ""
        echo "### [extra] Causal neuron ablation: $DATASET / N=$ABLATION_BUDGET (GPU, slow)"
        python refusal_direction/run_ablation.py \
            --prompts "cka/prompts/${DATASET}.csv" \
            --direction "$DIRECTION_NPZ" \
            --model_path "$MODEL_PATH" \
            --methods $METHODS \
            --budget "$ABLATION_BUDGET" \
            --n_harmful "$N_HARMFUL" \
            --n_benign "$N_BENIGN" \
            --null_seeds "$ABLATION_NULL_SEEDS" \
            --label "$LABEL"
    fi
done

echo ""
echo "Done. Results in refusal_direction/results/"
ls -1 refusal_direction/results/ | head -40
