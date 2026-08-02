#!/bin/bash
#SBATCH --job-name=siren-eval-std
#SBATCH --gpus=1                       # 1 GPU; no partition tag needed on ETH Euler
#SBATCH --gres=gpumem:20g              # 24GB cards are plenty for batch-8 extraction
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G               # 4 x 8G = 32G total
#SBATCH --time=02:00:00                # 200 test prompts, well under 2h
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Evaluate a trained SIREN model on the standardized HarmBench + Alpaca test
# split (100 harmful + 100 safe). Submit from repo root:
#   sbatch cluster/eval_standard_siren.sbatch llama3-8b-instruct
#   sbatch cluster/eval_standard_siren.sbatch llama3-8b-sft
#
# Requires train/probes/optuna/${MODEL}_general/best_model.pkl from the train job.

set -e
MODEL="${1:-llama3-8b-instruct}"

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

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL TEXT_COLUMN=${SIREN_TEXT_COLUMN:-formatted_input}"

cd test
bash eval_standard_siren.sh "$MODEL"

# Layer-wise analysis (probe F1 per layer + safety-neuron distribution)
cd "$SLURM_SUBMIT_DIR"
python analysis/plot_layer_probes.py --models "$MODEL" --threshold 0.9 || true
