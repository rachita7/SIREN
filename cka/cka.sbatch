#!/bin/bash
#SBATCH --job-name=cka
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g              # 24GB cards are enough at batch size 8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G              # 48G total: the activation array is ~2GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Submit from the repo root:
#   sbatch cka/cka.sbatch
#   DATASETS="wildguard xstest" sbatch --export=ALL,DATASETS cka/cka.sbatch
#
# Prompt sets are downloaded from HuggingFace, so run once on a login node with
# internet if your compute nodes are offline:
#   python cka/build_prompts.py --dataset wildguard
#   python cka/build_prompts.py --dataset xstest

set -e

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
export MPLBACKEND=Agg

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash cka/run_all.sh
