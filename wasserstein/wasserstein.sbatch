#!/bin/bash
#SBATCH --job-name=wasserstein
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G               # 48G total; activations ~2GB, cost matrix <1GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-only: reuses the activation tensors a cka/ job already extracted and
# fails fast if they are missing (SKIP_EXTRACT=1). Submit with the same
# RUN_TAG / DATASETS / METHODS / BUDGETS as the CKA job you want to compare to:
#
#   RUN_TAG=all7 DATASETS=wildguard METHODS=all BUDGETS="459 2294 4588 9175" \
#       sbatch --export=ALL,RUN_TAG,DATASETS,METHODS,BUDGETS wasserstein/wasserstein.sbatch
#
# Runtime at N=2294, 7 methods, 3 variants, 20 null seeds: ~10 min per
# dataset. N=9175 is ~30x slower per assignment (O(k^3)); reduce NULL_SEEDS
# there if the wall time is tight.

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if [ -z "${RUN_TAG:-}" ]; then
    echo "WARNING: RUN_TAG is unset; outputs are named by dataset+pooling only."
fi

export SKIP_EXTRACT="${SKIP_EXTRACT:-1}"
bash wasserstein/run_all.sh
