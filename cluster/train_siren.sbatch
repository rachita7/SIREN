#!/bin/bash
#SBATCH --job-name=siren-train
#SBATCH --gpus=1                       # 1 GPU; no partition tag needed on ETH Euler
#SBATCH --gres=gpumem:20g              # >=20GB GPU mem: opens RTX 3090/4090 (24GB), far more available than A100s
##SBATCH --account=es_yourgroup        # Euler: group share, NOT your username; usually auto-selected
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G              # 4 x 12G = 48G total (activations cache ~15G + overhead)
#SBATCH --time=18:00:00                # headroom for slower GPUs; keep <=24h (same Euler queue class)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Train SIREN on HH-RLHF. Submit from the repo root:
#   sbatch cluster/train_siren.sbatch llama3-8b-sft
#   sbatch cluster/train_siren.sbatch llama3-8b-instruct
#   sbatch cluster/train_siren.sbatch llama3-8b-sft mlpneuron_mean   # down_proj-input neurons

set -e
MODEL="${1:-llama3-8b-sft}"
REP_TYPE="${2:-residual_mean}"

# ETH Euler: eth_proxy gives compute nodes internet access (safe to keep even
# with the offline flags below; remove the line on non-ETH clusters).
module load eth_proxy 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siren

# Must match the HF_HOME used in cluster/setup_env.sh (Euler scratch if present)
if [ -d "/cluster/scratch/$USER" ]; then
    export HF_HOME="${HF_HOME:-/cluster/scratch/$USER/hf_cache}"
else
    export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
fi
# Online mode: eth_proxy gives Euler compute nodes internet, and offline mode
# breaks datasets' cache lookup for load_dataset(..., data_dir=...) (config-name
# hash mismatch). Downloads still hit the prefetched cache, so this is cheap.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

# Number of HH-RLHF preference pairs used for train+val (2 texts each)
export HH_RLHF_MAX_ROWS="${HH_RLHF_MAX_ROWS:-8000}"

# Extraction batch size 8 keeps peak GPU memory ~20GB so the job fits 24GB
# cards (raise to 16 if you request gpumem:40g instead).
export BATCH_SIZE="${BATCH_SIZE:-8}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=$MODEL REP_TYPE=$REP_TYPE HH_RLHF_MAX_ROWS=$HH_RLHF_MAX_ROWS"

cd train
bash run_hh_siren.sh "$MODEL" "$REP_TYPE"
