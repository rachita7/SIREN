#!/bin/bash
#SBATCH --job-name=siren-train-std
#SBATCH --gpus=1                       # 1 GPU; no partition tag needed on ETH Euler
#SBATCH --gres=gpumem:20g              # >=20GB GPU mem: opens RTX 3090/4090 (24GB)
##SBATCH --account=es_yourgroup        # Euler: group share, NOT your username
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G              # 4 x 12G = 48G total
#SBATCH --time=18:00:00                # headroom for slower GPUs; keep <=24h
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Train SIREN on the standardized HarmBench + Alpaca benchmark (data_files/).
# Submit from the repo root:
#   sbatch cluster/train_standard_siren.sbatch llama3-8b-instruct
#   sbatch cluster/train_standard_siren.sbatch llama3-8b-sft
#   sbatch cluster/train_standard_siren.sbatch llama3-8b-instruct mlpneuron_mean
#
# The train/val/test CSVs ship in the repo, so no dataset download is needed --
# only the backbone weights + tokenizer must already be in the HF cache
# (run cluster/download_assets.py on a login node first).

set -e
MODEL="${1:-llama3-8b-instruct}"
REP_TYPE="${2:-residual_mean}"

# ETH Euler: eth_proxy gives compute nodes internet access (harmless elsewhere)
module load eth_proxy 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

# Must match the HF_HOME used in cluster/setup_env.sh (Euler scratch if present)
if [ -d "/cluster/scratch/$USER" ]; then
    export HF_HOME="${HF_HOME:-/cluster/scratch/$USER/hf_cache}"
else
    export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

# Extraction batch size 8 keeps peak GPU memory ~20GB (fits 24GB cards).
export BATCH_SIZE="${BATCH_SIZE:-8}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL REP_TYPE=$REP_TYPE TEXT_COLUMN=${SIREN_TEXT_COLUMN:-formatted_input}"

cd train
bash run_standard_siren.sh "$MODEL" "$REP_TYPE"
