#!/bin/bash
#SBATCH --job-name=medsam3-eval
#SBATCH --partition=gpu            # adjust to your HPC partition name
#SBATCH --gres=gpu:1               # 1 GPU (A100/V100)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/medsam3_eval_%j.out
#SBATCH --error=logs/medsam3_eval_%j.err

# ── Adjust these paths ──────────────────────────────────────────────
PROJECT_DIR="/scratch/ll5582/Projects/MedSAM3"
SHARD_DIR="${PROJECT_DIR}/shards"        # where the .tar shards are
CHECKPOINT="${PROJECT_DIR}/checkpoints/checkpoint.pt"

PROMPT="lesion"
THRESHOLD=0.1
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate medsam3

cd "${PROJECT_DIR}"

echo "=== Medical SAM3 Evaluation ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Shards: ${SHARD_DIR}"
echo "Prompt: ${PROMPT}"
echo "Threshold: ${THRESHOLD}"
echo ""

python evaluate_webdataset.py \
    --shards "${SHARD_DIR}" \
    --prompt "${PROMPT}" \
    --threshold "${THRESHOLD}" \
    --checkpoint "${CHECKPOINT}" \
    --viz-dir results/viz \
    --output results/eval_results.json

echo ""
echo "=== Done ==="
