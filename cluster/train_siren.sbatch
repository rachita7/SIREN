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
#
# mlpneuron_mean (14336-dim, ~3.5x residual_mean's memory) needs more RAM:
#   - threshold 0.9 selects ~266k features and OOMs stage 3 below ~96GB total,
#     so either drop it (THRESHOLDS="0.6 0.8") or submit with --mem-per-cpu=24G.
# C_VALUES / THRESHOLDS / OUTPUT_SUFFIX are read from the submit-time env
# (sbatch exports it to the job by default), e.g. a C-tuning run:
#   C_VALUES="1000.0 5000.0 20000.0" THRESHOLDS="0.6 0.8" \
#     OUTPUT_SUFFIX="-mlpneuron_mean-c20k" \
#     sbatch --mem-per-cpu=16G cluster/train_siren.sbatch llama3-8b-instruct mlpneuron_mean

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
echo "C_VALUES=${C_VALUES:-<default>} THRESHOLDS=${THRESHOLDS:-<default>} OUTPUT_SUFFIX=${OUTPUT_SUFFIX-<auto>}"

cd train
bash run_hh_siren.sh "$MODEL" "$REP_TYPE"
