#!/bin/bash
#SBATCH --job-name=siren-train-paper
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g              # 24GB cards; raise to 39g + BATCH_SIZE=16 for A100
##SBATCH --account=es_yourgroup
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Train SIREN on a single paper dataset. Submit from the repo root:
#   sbatch cluster/train_paper_siren.sbatch llama3-8b-instruct aegis2
#   sbatch cluster/train_paper_siren.sbatch llama3-8b-instruct wildguard
#
# Existing probes for this MODEL (e.g. from the HH-RLHF run) are archived to a
# timestamped backup before training, so previous results are not overwritten.

set -e
MODEL="${1:-llama3-8b-instruct}"
DATASET="${2:-aegis2}"
REP_TYPE="${3:-residual_mean}"

module load eth_proxy 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

if [ -d "/cluster/scratch/$USER" ]; then
    export HF_HOME="${HF_HOME:-/cluster/scratch/$USER/hf_cache}"
else
    export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

export BATCH_SIZE="${BATCH_SIZE:-8}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Archive any existing probes for this model so the HH-RLHF run is preserved.
TS=$(date +%Y%m%d_%H%M%S)
if [ -d "train/probes/optuna/${MODEL}_general" ]; then
    mkdir -p "train/probes/archive_${TS}/optuna"
    mv "train/probes/optuna/${MODEL}_general" "train/probes/archive_${TS}/optuna/"
    echo "Archived previous optuna results to train/probes/archive_${TS}/optuna/${MODEL}_general"
fi
if [ -f "train/probes/${MODEL}_general_probes.pkl" ]; then
    mkdir -p "train/probes/archive_${TS}"
    mv "train/probes/${MODEL}_general_probes.pkl" "train/probes/archive_${TS}/"
    echo "Archived previous probes to train/probes/archive_${TS}/${MODEL}_general_probes.pkl"
fi

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL DATASET=$DATASET REP_TYPE=$REP_TYPE BATCH_SIZE=$BATCH_SIZE"

cd train
bash run_paper_siren.sh "$MODEL" "$DATASET" "$REP_TYPE"
