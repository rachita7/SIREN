#!/bin/bash
#SBATCH --job-name=siren-eval
#SBATCH --partition=gpu                # TODO: your GPU partition
##SBATCH --account=YOUR_ACCOUNT        # TODO: uncomment + set if required
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Evaluate a trained SIREN model on HarmBench + AdvBench. Submit from repo root:
#   sbatch cluster/eval_siren.sbatch llama3-8b-sft
#   sbatch cluster/eval_siren.sbatch llama3-8b-instruct
#
# Requires train/probes/optuna/${MODEL}_general/best_model.pkl from the train job.

set -e
MODEL="${1:-llama3-8b-sft}"

# TODO: adjust to your cluster's module system
# module load miniconda3
# module load cuda/12.6

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# 1 = mix in equal number of safe Alpaca prompts as negatives (F1 meaningful)
# 0 = harmful prompts only (read recall as the detection rate)
export HARM_EVAL_ADD_SAFE="${HARM_EVAL_ADD_SAFE:-1}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL HARM_EVAL_ADD_SAFE=$HARM_EVAL_ADD_SAFE"

cd test
bash eval_harm_siren.sh "$MODEL"

# Layer-wise analysis (probe F1 per layer + safety-neuron distribution)
cd "$SLURM_SUBMIT_DIR"
python analysis/plot_layer_probes.py --models "$MODEL" --threshold 0.9 || true
