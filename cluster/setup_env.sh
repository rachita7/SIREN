#!/bin/bash
# One-time setup on the cluster LOGIN node (needs internet access).
#
# Usage:
#   export HF_TOKEN=hf_...        # required for meta-llama/Meta-Llama-3-8B-Instruct (gated)
#   bash cluster/setup_env.sh
#
# Adjust the module loads / paths marked TODO for your cluster.

set -e

# ETH Euler: login nodes have internet (no eth_proxy needed here). Conda is not
# a module on Euler; install Miniconda ONCE into $HOME before running this script:
#
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
#   $HOME/miniconda3/bin/conda init bash && source ~/.bashrc
#   conda config --set auto_activate_base false
#   # keep downloaded packages off $HOME's file quota:
#   conda config --add pkgs_dirs /cluster/scratch/$USER/conda_pkgs
#
# On other clusters, load your conda/cuda modules here instead, e.g.:
# module load miniconda3
# module load cuda/12.6

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Install Miniconda first (see comments above)." >&2
    exit 1
fi

# Keep the (large) HF cache off $HOME. On Euler /cluster/scratch/$USER is used
# automatically. NOTE: Euler scratch is purged after ~15 days of file age; if
# jobs later fail with missing weights, just re-run cluster/download_assets.py.
if [ -d "/cluster/scratch/$USER" ]; then
    export HF_HOME="${HF_HOME:-/cluster/scratch/$USER/hf_cache}"
else
    export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
fi
mkdir -p "$HF_HOME"
# Keep pip's cache off $HOME too (used during conda env create's pip stage)
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HF_HOME/../pip_cache}"

# This script lives in the repo (cluster/), so the repo root is one level up.
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
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
