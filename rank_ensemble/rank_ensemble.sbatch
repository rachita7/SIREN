#!/bin/bash
#SBATCH --job-name=rank-ensemble
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the repo root:
#   sbatch rank_ensemble/rank_ensemble.sbatch                          # all7, N=2294
#   BUDGETS="459 2294 4588 9175" sbatch --export=ALL,BUDGETS \
#       rank_ensemble/rank_ensemble.sbatch                             # all7, all N
#   EXPERIMENT=siren_yang_wang_zhao BUDGETS="459 2294 4588 9175" \
#       sbatch --export=ALL,EXPERIMENT,BUDGETS rank_ensemble/rank_ensemble.sbatch
#   EXPERIMENT=siren_yang_harm BUDGETS="459 2294 4588 9175" \
#       sbatch --export=ALL,EXPERIMENT,BUDGETS rank_ensemble/rank_ensemble.sbatch
#
# MMLU is downloaded from Hugging Face. On offline compute nodes, warm the
# cache on a login node first:
#   python -c "from datasets import load_dataset; load_dataset('cais/mmlu','all',split='test')"

set -e

module load eth_proxy 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

if [ -d "/cluster/scratch/$USER" ]; then
    export HF_HOME="${HF_HOME:-/cluster/scratch/$USER/hf_cache}"
else
    export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
fi
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash rank_ensemble/run_all.sh
