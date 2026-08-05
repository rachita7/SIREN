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

# On cluster login nodes `python` is only available inside the conda env.
if ! command -v python >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate siren 2>/dev/null || true
fi
PYTHON="$(command -v python || command -v python3)"

# mlpneuron_mean activations are 14336-dim vs residual_mean's 4096-dim
# (~3.5x GPU memory per sample): batch 8 fits a 24GB card.
export BATCH_SIZE="${BATCH_SIZE:-8}"

# Must match run_hh_siren.sh's OUTPUT_SUFFIX for the run being post-processed
# (default for a plain mlpneuron_mean run is "-mlpneuron_mean").
export OUTPUT_SUFFIX="${OUTPUT_SUFFIX:--mlpneuron_mean}"

PROBES_SRC="train/probes/${MODEL}_general_probes${OUTPUT_SUFFIX}.pkl"
RESULTS_SRC="train/probes/optuna/${MODEL}_general${OUTPUT_SUFFIX}/results.json"
PROBES_DST="results/${MODEL}_general_probes-hhrlhf${OUTPUT_SUFFIX}.pkl"
JSON_DST="results/SIREN_detected_Neurons${OUTPUT_SUFFIX}.json"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    ( cd train && bash run_hh_siren.sh "$MODEL" mlpneuron_mean )
fi

cp "$PROBES_SRC" "$PROBES_DST"
cp "$RESULTS_SRC" "$JSON_DST"
echo ""
echo "Snapshotted probes -> $PROBES_DST"
echo "Exported JSON      -> $JSON_DST   <-- send this file"

echo ""
echo "Rendering FFN-space overlap plots (SIREN mlpneuron vs zhao/svea ffn)..."
"$PYTHON" analysis/plot_ffn_space_overlap.py \
    --siren_probes "$PROBES_DST" \
    --svea results/probes-svea.pkl \
    --threshold "${PLOT_THRESHOLD:-0.9}"

echo ""
echo "Optional: evaluate on the standardized HarmBench+Alpaca split with:"
echo "  cd test && bash eval_standard_siren.sh ${MODEL}"
