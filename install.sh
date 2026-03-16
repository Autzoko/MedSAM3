#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="medsam3"
PYTHON_VER="3.12"

echo "=== Medical SAM3 Web Demo — Installer ==="

# ── 1. Conda environment ────────────────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists — activating."
else
    echo "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VER})…"
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
fi

# Activate inside the script
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Python: $(python --version)  ($(which python))"

# ── 2. PyTorch (CPU / MPS — no CUDA on macOS) ──────────────────────
echo "Installing PyTorch 2.7+ (CPU/MPS)…"
pip install --upgrade pip
pip install 'torch>=2.7' 'torchvision>=0.22' 'torchaudio>=2.7'

# ── 3. Base SAM3 library ────────────────────────────────────────────
SAM3_DIR="$(dirname "$0")/third_party/sam3"
if [ -d "${SAM3_DIR}" ]; then
    echo "SAM3 repo already cloned."
else
    echo "Cloning facebook/sam3…"
    mkdir -p "$(dirname "$0")/third_party"
    git clone https://github.com/facebookresearch/sam3.git "${SAM3_DIR}"
fi
echo "Installing SAM3 in editable mode…"
pip install -e "${SAM3_DIR}"

# SAM3 has many transitive deps not listed in its core requirements.
# Install everything needed for the import chain to succeed on macOS.
echo "Installing SAM3 transitive dependencies…"
pip install einops pycocotools psutil scikit-learn scikit-image \
    opencv-python matplotlib pandas scipy regex ftfy

# ── 4. Medical-SAM3 repo ────────────────────────────────────────────
MEDSAM3_DIR="$(dirname "$0")/third_party/Medical-SAM3"
if [ -d "${MEDSAM3_DIR}" ]; then
    echo "Medical-SAM3 repo already cloned."
else
    echo "Cloning Medical-SAM3…"
    git clone https://github.com/AIM-Research-Lab/Medical-SAM3.git "${MEDSAM3_DIR}"
fi

# Install its requirements (skip torch — already installed)
if [ -f "${MEDSAM3_DIR}/requirements.txt" ]; then
    echo "Installing Medical-SAM3 requirements…"
    pip install -r "${MEDSAM3_DIR}/requirements.txt" || true
fi

# ── 5. Web-demo dependencies ────────────────────────────────────────
echo "Installing web-demo Python deps…"
pip install fastapi uvicorn python-multipart pillow numpy huggingface_hub tqdm

# Note: 'decord' (video reader) and 'triton' (GPU kernels) are not
# available on macOS. They are stubbed out at runtime — only needed
# for video tracking / CUDA paths, not for image inference.

# ── 6. Done ─────────────────────────────────────────────────────────
echo ""
echo "✓ Installation complete."
echo "  Next steps:"
echo "    conda activate ${ENV_NAME}"
echo "    python download_ckpt.py          # download the 10 GB checkpoint"
echo "    python backend/app.py            # start the server on :8000"
