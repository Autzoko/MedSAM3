#!/bin/bash
#SBATCH --job-name=medsam3-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1               # 1 GPU (A100 recommended)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/medsam3_finetune_%j.out
#SBATCH --error=logs/medsam3_finetune_%j.err

# ── Adjust these paths ──────────────────────────────────────────────
PROJECT_DIR="/scratch/ll5582/Projects/MedSAM3"
SHARD_DIR="${PROJECT_DIR}/shards_abus"
CHECKPOINT="${PROJECT_DIR}/checkpoints/checkpoint.pt"
OUTPUT_DIR="${PROJECT_DIR}/finetune_output"

PROMPT="breast lesion"
EPOCHS=20
BATCH_SIZE=1
LR=3e-4
LR_VISION=5e-5
LR_LANGUAGE=5e-5
LR_GEOMETRY=1e-4
LRD=0.85
WARMUP=50
VAL_FREQ=2
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate medsam3

cd "${PROJECT_DIR}"

echo "=== Medical SAM3 Finetuning ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Shards: ${SHARD_DIR}"
echo "Prompt: ${PROMPT}"
echo "Epochs: ${EPOCHS}, BS: ${BATCH_SIZE}, LR: ${LR}"
echo ""

python finetune_abus.py \
    --shards "${SHARD_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --text-prompt "${PROMPT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr-vision "${LR_VISION}" \
    --lr-language "${LR_LANGUAGE}" \
    --lr-geometry "${LR_GEOMETRY}" \
    --lrd "${LRD}" \
    --warmup-steps "${WARMUP}" \
    --val-freq "${VAL_FREQ}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers 4

echo ""
echo "=== Finetuning Done ==="
echo "Checkpoints: ${OUTPUT_DIR}/"

# Optionally run evaluation with the best checkpoint
echo ""
echo "=== Running Evaluation with Best Checkpoint ==="
python evaluate_webdataset.py \
    --shards "${PROJECT_DIR}/shards" \
    --prompt "${PROMPT}" \
    --threshold 0.1 \
    --checkpoint "${OUTPUT_DIR}/best_checkpoint.pt" \
    --viz-dir "${OUTPUT_DIR}/viz" \
    --output "${OUTPUT_DIR}/eval_results.json"

echo "=== All Done ==="
