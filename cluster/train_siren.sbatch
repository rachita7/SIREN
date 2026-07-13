#!/bin/bash
#SBATCH --job-name=siren-train
#SBATCH --gpus=1                       # 1 GPU; no partition tag needed on ETH Euler
#SBATCH --gres=gpumem:40g              # require >=40GB GPU memory (80GB comfortable)
##SBATCH --account=es_yourgroup        # Euler: group share, NOT your username; usually auto-selected
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G              # 8 x 16G = 128G total
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Train SIREN on HH-RLHF. Submit from the repo root:
#   sbatch cluster/train_siren.sbatch llama3-8b-sft
#   sbatch cluster/train_siren.sbatch llama3-8b-instruct
#   sbatch cluster/train_siren.sbatch llama3-8b-sft mlpneuron_mean   # down_proj-input neurons

set -e
MODEL="${1:-llama3-8b-sft}"
REP_TYPE="${2:-residual_mean}"

# ETH Euler: eth_proxy gives compute nodes internet access (safe to keep even
# with the offline flags below; remove the line on non-ETH clusters).
module load eth_proxy 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

# Must match the HF_HOME used in cluster/setup_env.sh
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # compute nodes read the prefetched cache
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# Number of HH-RLHF preference pairs used for train+val (2 texts each)
export HH_RLHF_MAX_ROWS="${HH_RLHF_MAX_ROWS:-8000}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL REP_TYPE=$REP_TYPE HH_RLHF_MAX_ROWS=$HH_RLHF_MAX_ROWS"

cd train
bash run_hh_siren.sh "$MODEL" "$REP_TYPE"
