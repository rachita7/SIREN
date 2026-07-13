#!/bin/bash
#SBATCH --job-name=siren-train-fast
#SBATCH --gpus=1
#SBATCH --gres=gpumem:39g              # excludes 24GB cards -> lands on A100 40/80GB (or better)
##SBATCH --account=es_yourgroup        # Euler: group share, usually auto-selected
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G              # 4 x 12G = 48G total
#SBATCH --time=10:00:00                # A100 is ~2-3x faster than the 24GB cards
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fast-GPU variant of train_siren.sbatch: same pipeline, bigger extraction
# batch, targets A100-class GPUs. Race it against the regular script:
#   sbatch cluster/train_siren.sbatch      llama3-8b-instruct
#   sbatch cluster/train_siren_fast.sbatch llama3-8b-instruct
# IMPORTANT: both write to the same output paths -- as soon as one job starts
# running (squeue --me), scancel the other.

set -e
MODEL="${1:-llama3-8b-sft}"
REP_TYPE="${2:-residual_mean}"

# ETH Euler: internet access on compute nodes
module load eth_proxy 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

# Must match the HF_HOME used in cluster/setup_env.sh (Euler scratch if present)
if [ -d "/cluster/scratch/$USER" ]; then
    export HF_HOME="${HF_HOME:-/cluster/scratch/$USER/hf_cache}"
else
    export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
fi
# Online mode: see note in train_siren.sbatch (offline breaks data_dir cache lookup)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

# Number of HH-RLHF preference pairs used for train+val (2 texts each)
export HH_RLHF_MAX_ROWS="${HH_RLHF_MAX_ROWS:-8000}"

# 40GB+ GPU comfortably fits batch 16
export BATCH_SIZE="${BATCH_SIZE:-16}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL REP_TYPE=$REP_TYPE HH_RLHF_MAX_ROWS=$HH_RLHF_MAX_ROWS BATCH_SIZE=$BATCH_SIZE"

cd train
bash run_hh_siren.sh "$MODEL" "$REP_TYPE"
