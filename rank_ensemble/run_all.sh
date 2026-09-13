#!/bin/bash
# Rank-ensemble driver. Run from the repo root:
#   bash rank_ensemble/run_all.sh
#   BUDGETS="459 2294 4588 9175" bash rank_ensemble/run_all.sh
#   EXPERIMENT=siren_yang_wang_zhao BUDGETS="459 2294 4588 9175" bash rank_ensemble/run_all.sh
#   EXPERIMENT=siren_yang_harm BUDGETS="459 2294 4588 9175" bash rank_ensemble/run_all.sh
#
# EXPERIMENT=all7 (default) writes to rank_ensemble/{selections,results}
# Other names write to rank_ensemble/experiments/<name>/ and never touch all7.
set -e
cd "$(dirname "$0")/.."

EXPERIMENT="${EXPERIMENT:-all7}"
BUDGETS="${BUDGETS:-2294}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
TARGET_NAME="${TARGET_NAME:-}"
N_MMLU="${N_MMLU:-2000}"
N_ADVBENCH="${N_ADVBENCH:-100}"

echo "[1/3] validate + build ensembles  experiment=${EXPERIMENT}"
python rank_ensemble/build_ensembles.py --experiment "$EXPERIMENT" --budgets $BUDGETS

extra=()
[ -n "$LORA_ADAPTER" ] && extra+=(--lora-adapter "$LORA_ADAPTER")
[ -n "$TARGET_NAME" ] && extra+=(--target-name "$TARGET_NAME")

for n in $BUDGETS; do
    echo "[2/3] ablation eval at N=$n  experiment=${EXPERIMENT}"
    python rank_ensemble/run_ablation_target.py --experiment "$EXPERIMENT" \
        --budget "$n" \
        --n-advbench-prompts "$N_ADVBENCH" --n-mmlu-questions "$N_MMLU" \
        "${extra[@]}"
done

echo "[3/3] tables + plots  experiment=${EXPERIMENT}"
python rank_ensemble/analyze_results.py --experiment "$EXPERIMENT" --budgets $BUDGETS
echo "Done. Experiment ${EXPERIMENT} results under rank_ensemble/ (see printed results_dir)"
