#!/usr/bin/env bash
set -euo pipefail

# ── Install Medical SAM3 on NYU Jubail HPC ─────────────────────────
# Run this ONCE from a login node (or inside an interactive session).
# Assumes: conda/mamba is available via `module load`

ENV_NAME="medsam3"
PYTHON_VER="3.12"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Medical SAM3 HPC Setup ==="
echo "Project dir: ${PROJECT_DIR}"

# ── 1. Conda env ────────────────────────────────────────────────────
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Env '${ENV_NAME}' exists."
else
    echo "Creating env..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ── 2. PyTorch with CUDA ────────────────────────────────────────────
echo "Installing PyTorch (CUDA 12.6)..."
pip install --upgrade pip
pip install 'torch>=2.7' 'torchvision>=0.22' 'torchaudio>=2.7' \
    --index-url https://download.pytorch.org/whl/cu126

# ── 3. SAM3 + deps ─────────────────────────────────────────────────
SAM3_DIR="${PROJECT_DIR}/third_party/sam3"
MEDSAM3_DIR="${PROJECT_DIR}/third_party/Medical-SAM3"

if [ ! -d "${SAM3_DIR}" ]; then
    git clone https://github.com/facebookresearch/sam3.git "${SAM3_DIR}"
fi
pip install -e "${SAM3_DIR}"

if [ ! -d "${MEDSAM3_DIR}" ]; then
    git clone https://github.com/AIM-Research-Lab/Medical-SAM3.git "${MEDSAM3_DIR}"
fi
[ -f "${MEDSAM3_DIR}/requirements.txt" ] && pip install -r "${MEDSAM3_DIR}/requirements.txt" || true

# SAM3 transitive deps
pip install einops pycocotools psutil scikit-learn scikit-image \
    opencv-python-headless matplotlib pandas scipy regex ftfy triton

# Finetuning deps (Hungarian matching, schedulers)
pip install fvcore hydra-core omegaconf submitit

# Web demo + eval deps
pip install fastapi uvicorn python-multipart pillow numpy huggingface_hub tqdm

echo ""
echo "✓ Done. Next: download checkpoint and submit jobs."
echo "  python download_ckpt.py"
echo "  sbatch hpc/run_eval.slurm       # evaluation"
echo "  sbatch hpc/run_finetune.slurm   # finetuning"
