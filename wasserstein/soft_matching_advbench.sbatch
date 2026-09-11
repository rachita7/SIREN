#!/bin/bash
#SBATCH --job-name=sm-advbench
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Soft Matching on AdvBench, top 1% (N=4588), all 7 methods.
# CPU only. Requires cka/activations/advbench_mean.npy already extracted.
#
#   sbatch wasserstein/soft_matching_advbench.sbatch
#
# Writes only:
#   wasserstein/results/soft_matching_advbench_N4588.png
#   wasserstein/results/soft_matching_advbench_N4588.csv

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

ACTS="${ACTS:-cka/activations/advbench_mean.npy}"
if [ ! -f "$ACTS" ]; then
    echo "ERROR: $ACTS not found. Extract AdvBench activations first"
    echo "  (wasserstein/wasserstein_advbench.sbatch, or cka/extract_activations.py)."
    exit 1
fi

python wasserstein/run_soft_matching.py --activations "$ACTS"
