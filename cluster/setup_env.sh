#!/bin/bash
# One-time setup on the cluster LOGIN node (needs internet access).
#
# Usage:
#   export HF_TOKEN=hf_...        # required for meta-llama/Meta-Llama-3-8B-Instruct (gated)
#   bash cluster/setup_env.sh
#
# Adjust the module loads / paths marked TODO for your cluster.

set -e

# ETH Euler: login nodes have internet, no eth_proxy needed here. Conda is not
# a module on Euler; install miniconda into your $HOME or scratch first if you
# don't have it (https://docs.conda.io/en/latest/miniconda.html).
# On other clusters, load your conda/cuda modules here, e.g.:
# module load miniconda3
# module load cuda/12.6

# Keep the (large) HF cache on scratch/project storage, not in $HOME.
# TODO: point this at your scratch space.
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME"

REPO_URL="https://github.com/rachita7/SIREN.git"
REPO_DIR="${REPO_DIR:-$HOME/SIREN}"

if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

if ! conda env list | grep -q "^siren "; then
    conda env create -f environment.yaml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set."
    echo "meta-llama/Meta-Llama-3-8B-Instruct is gated: request access on its HF page,"
    echo "then 'export HF_TOKEN=hf_...' and re-run this script."
else
    hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
        || huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Prefetch datasets + model weights so compute nodes (often offline) only read the cache.
python cluster/download_assets.py

echo ""
echo "Setup complete. Persist HF_HOME in your ~/.bashrc:"
echo "  export HF_HOME=$HF_HOME"
echo ""
echo "Submit jobs from the repo root, e.g.:"
echo "  sbatch cluster/train_siren.sbatch llama3-8b-sft"
echo "  sbatch cluster/train_siren.sbatch llama3-8b-instruct"
