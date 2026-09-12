#!/bin/bash
# Rank-ensemble driver. Run from the repo root:
#   bash rank_ensemble/run_all.sh
#   BUDGETS="459 2294 4588 9175" bash rank_ensemble/run_all.sh
#   BUDGETS="2294" TARGETS="rank_consensus quota" bash rank_ensemble/run_all.sh
#   LORA_ADAPTER=/path/to/dpo_adapter bash rank_ensemble/run_all.sh
set -e
cd "$(dirname "$0")/.."

BUDGETS="${BUDGETS:-2294}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
TARGET_NAME="${TARGET_NAME:-}"
N_MMLU="${N_MMLU:-2000}"
N_ADVBENCH="${N_ADVBENCH:-100}"

echo "[1/3] validate + build ensembles"
python rank_ensemble/build_ensembles.py --budgets $BUDGETS

extra=()
[ -n "$LORA_ADAPTER" ] && extra+=(--lora-adapter "$LORA_ADAPTER")
[ -n "$TARGET_NAME" ] && extra+=(--target-name "$TARGET_NAME")

for n in $BUDGETS; do
    echo "[2/3] ablation eval at N=$n"
    python rank_ensemble/run_ablation_target.py --budget "$n" \
        --n-advbench-prompts "$N_ADVBENCH" --n-mmlu-questions "$N_MMLU" \
        "${extra[@]}"
done

echo "[3/3] tables + plots"
python rank_ensemble/analyze_results.py --budgets $BUDGETS
echo "Done. Results in rank_ensemble/results/"
