#!/bin/bash
#SBATCH --job-name=cka-analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G               # 48G total; the activation array is ~2GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-ONLY rerun of the analysis steps on activations a previous job already
# extracted. No GPU is requested, so this queues far faster than cka.sbatch and
# can run alongside another job.
#
# Always set RUN_TAG, or this will overwrite the earlier run's outputs:
#   RUN_TAG=all8 DATASETS=wildguard METHODS=all BUDGETS="2500 5000 10000" \
#       sbatch --export=ALL,RUN_TAG,DATASETS,METHODS,BUDGETS cka/cka_analysis.sbatch
#
# It fails immediately if the activations are missing rather than starting a
# duplicate extraction, so it is safe to submit while the GPU job is still
# extracting -- though you then have to resubmit once that finishes.

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if [ -z "${RUN_TAG:-}" ]; then
    echo "WARNING: RUN_TAG is unset. If another run used the same dataset and"
    echo "pooling, its CSVs and figures will be overwritten."
fi

export SKIP_EXTRACT=1
bash cka/run_all.sh
