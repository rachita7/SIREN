#!/bin/bash
#SBATCH --job-name=siren-clean-stability
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00                # 3 small probe-only runs (~173 texts each)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# SIREN clean-stability protocol: 3 probe trainings on the pre-made thirds of
# the cleaned train set. Submit from the repo root:
#   sbatch cluster/train_clean_stability_siren.sbatch llama3-8b-instruct mlpneuron_mean

set -e
MODEL="${1:-llama3-8b-instruct}"
REP_TYPE="${2:-mlpneuron_mean}"

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

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL REP_TYPE=$REP_TYPE"

cd train
bash run_clean_stability_siren.sh "$MODEL" "$REP_TYPE"
