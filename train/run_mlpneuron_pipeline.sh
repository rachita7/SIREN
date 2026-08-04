#!/bin/bash
# End-to-end pipeline for the FFN-space (mlpneuron_mean) SIREN run:
#   1. train SIREN probes on the down_proj-input space (14336-dim MLP neurons)
#   2. snapshot the artifacts into results/ with a -mlpneuron suffix
#   3. export the JSON to send for the cross-method comparison
#      (results.json already has the right shape: best_overall + all_results,
#      final_mlp already stripped)
#   4. render FFN-space overlap plots vs the Zhao/Svea ffn selection
#
# Run from anywhere:
#   bash train/run_mlpneuron_pipeline.sh [MODEL]        # default llama3-8b-instruct
#
# On the cluster, submit training separately with extra RAM (the 14336-dim
# activation cache is ~3.5x the residual_mean one, ~50GB+):
#   sbatch --mem-per-cpu=16G cluster/train_siren.sbatch llama3-8b-instruct mlpneuron_mean
# then run only the post-processing here:
#   SKIP_TRAIN=1 bash train/run_mlpneuron_pipeline.sh llama3-8b-instruct

set -e
MODEL="${1:-llama3-8b-instruct}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# mlpneuron_mean activations are 14336-dim vs residual_mean's 4096-dim
# (~3.5x GPU memory per sample): batch 8 fits a 24GB card.
export BATCH_SIZE="${BATCH_SIZE:-8}"

PROBES_SRC="train/probes/${MODEL}_general_probes.pkl"
RESULTS_SRC="train/probes/optuna/${MODEL}_general/results.json"
PROBES_DST="results/${MODEL}_general_probes-hhrlhf-mlpneuron.pkl"
JSON_DST="results/SIREN_detected_Neurons_mlpneuron.json"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    # Training overwrites the generic probes pkl; keep the previous
    # (residual_mean) run around just in case.
    if [ -f "$PROBES_SRC" ]; then
        cp "$PROBES_SRC" "${PROBES_SRC%.pkl}.pre-mlpneuron-backup.pkl"
    fi
    ( cd train && bash run_hh_siren.sh "$MODEL" mlpneuron_mean )
fi

cp "$PROBES_SRC" "$PROBES_DST"
cp "$RESULTS_SRC" "$JSON_DST"
echo ""
echo "Snapshotted probes -> $PROBES_DST"
echo "Exported JSON      -> $JSON_DST   <-- send this file"

echo ""
echo "Rendering FFN-space overlap plots (SIREN mlpneuron vs zhao/svea ffn)..."
python analysis/plot_ffn_space_overlap.py \
    --siren_probes "$PROBES_DST" \
    --svea results/probes-svea.pkl \
    --threshold "${PLOT_THRESHOLD:-0.9}"

echo ""
echo "Optional: evaluate on the standardized HarmBench+Alpaca split with:"
echo "  cd test && bash eval_standard_siren.sh ${MODEL}"
