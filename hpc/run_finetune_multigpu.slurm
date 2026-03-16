#!/bin/bash
#SBATCH --job-name=medsam3-ft-mgpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # torchrun handles spawning
#SBATCH --cpus-per-task=32             # 8 per GPU × 4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/medsam3_ft_mgpu_%j.out
#SBATCH --error=logs/medsam3_ft_mgpu_%j.err

# ── GPU count: set to 2 or 4 ─────────────────────────────────────
#SBATCH --gres=gpu:4                   # ← change to gpu:2 for 2-GPU
NUM_GPUS=4                             # ← must match --gres above

# ── Adjust these paths ──────────────────────────────────────────────
PROJECT_DIR="/scratch/ll5582/Projects/MedSAM3"
SHARD_DIR="${PROJECT_DIR}/shards_abus"
CHECKPOINT="${PROJECT_DIR}/checkpoints/checkpoint.pt"
OUTPUT_DIR="${PROJECT_DIR}/finetune_output_${NUM_GPUS}gpu"

PROMPT="breast lesion"
EPOCHS=20
BATCH_SIZE=1                           # per-GPU batch size
ACCUM_STEPS=1                          # gradient accumulation (effective BS = BS × ACCUM × NGPU)
LR=1e-4
LR_VISION=1e-5
LR_LANGUAGE=1e-5
LR_GEOMETRY=3e-5
LRD=0.85
WARMUP=200
VAL_FREQ=2
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate medsam3

cd "${PROJECT_DIR}"

echo "=== Medical SAM3 Multi-GPU Finetuning ==="
echo "Node:     $(hostname)"
echo "GPUs:     ${NUM_GPUS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo "Shards:   ${SHARD_DIR}"
echo "Prompt:   ${PROMPT}"
echo "Epochs:   ${EPOCHS}, per-GPU BS: ${BATCH_SIZE}, accum: ${ACCUM_STEPS}"
echo "Eff. BS:  $((BATCH_SIZE * ACCUM_STEPS * NUM_GPUS))"
echo ""

# NCCL tuning for multi-GPU on single node
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    finetune_abus.py \
    --shards "${SHARD_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --text-prompt "${PROMPT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --accum-steps "${ACCUM_STEPS}" \
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

# Evaluation with best checkpoint (single-GPU is fine for eval)
echo ""
echo "=== Running Evaluation with Best Checkpoint ==="
python evaluate_webdataset.py \
    --shards "${SHARD_DIR}" \
    --prompt "${PROMPT}" \
    --threshold 0.1 \
    --checkpoint "${OUTPUT_DIR}/best_checkpoint.pt" \
    --viz-dir "${OUTPUT_DIR}/viz" \
    --output "${OUTPUT_DIR}/eval_results.json"

echo "=== All Done ==="
