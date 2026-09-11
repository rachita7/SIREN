#!/bin/bash
#SBATCH --job-name=wd-advbench
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Wasserstein on AdvBench only. Every activation and result file is tagged
# `advbench`, so this cannot overwrite a wildguard / xstest run.
#
# Submit from the repo root, even while the wildguard job is still running:
#   sbatch wasserstein/wasserstein_advbench.sbatch
#   METHODS=all sbatch --export=ALL,METHODS wasserstein/wasserstein_advbench.sbatch
#
# If compute nodes cannot reach GitHub, download the prompt CSV first on a
# login node (official llm-attacks CSV, no HuggingFace access request):
#   python cka/build_prompts.py --dataset advbench --max_prompts 2000
#
# Outputs (will not collide with wildguard):
#   cka/prompts/advbench.csv
#   cka/activations/advbench_mean.npy
#   wasserstein/results/wd_advbench_mean_N2294.csv
#   wasserstein/results/wd_vs_cka_advbench_mean_N2294.csv
#   wasserstein/results/wd_*_advbench_mean_*
#
# AdvBench is harmful-only (~520 prompts). The `class` residualization is
# uninformative (one label); read `raw` and `class+length` (length still
# varies). With fewer prompts than neurons, use z / normalized, not raw W.

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

# Pin the dataset even if the submitting shell exported DATASETS=wildguard.
export DATASETS=advbench
export SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
export SKIP_LAYER_W1=1
# Do not inherit a wildguard RUN_TAG; files are already named advbench_*.
export RUN_TAG="${WD_ADVBENCH_TAG:-}"

echo "Job $SLURM_JOB_ID on $(hostname)  DATASETS=$DATASETS  METHODS=${METHODS:-default}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash wasserstein/run_all.sh
