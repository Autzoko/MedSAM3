#!/bin/bash
#SBATCH --job-name=medsam3-infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1               # 1 GPU is sufficient for inference
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/medsam3_infer_%j.out
#SBATCH --error=logs/medsam3_infer_%j.err

# ── Adjust these paths ──────────────────────────────────────────────
PROJECT_DIR="/scratch/ll5582/Projects/MedSAM3"
SHARD_DIR="${PROJECT_DIR}/shards_birads"
CHECKPOINT="${PROJECT_DIR}/finetune_output_2gpu/best_checkpoint.pt"
OUTPUT_DIR="${PROJECT_DIR}/infer_results"

PROMPT="breast lesion"
THRESHOLD=0.1
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs "${OUTPUT_DIR}"

module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate medsam3

cd "${PROJECT_DIR}"

echo "=== Medical SAM3 BIRADS Inference ==="
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Shards:     ${SHARD_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Prompt:     ${PROMPT}"
echo ""

python infer_webdataset.py \
    --shards "${SHARD_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --prompt "${PROMPT}" \
    --threshold "${THRESHOLD}" \
    --output "${OUTPUT_DIR}/infer_results.json" \
    --viz-dir "${OUTPUT_DIR}/viz"

echo ""
echo "=== Inference Done ==="
echo "Results: ${OUTPUT_DIR}/"
