#!/usr/bin/env python3
"""FastAPI server for the Medical SAM3 web demo.

Run:
    python backend/app.py          # starts on http://localhost:8000
    MEDSAM3_PORT=9000 python backend/app.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# MPS fallback for ops not yet implemented on Apple GPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND = _ROOT / "frontend"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("medsam3")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Medical SAM3 Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Suggested prompts (users can also type free-form)
# ---------------------------------------------------------------------------
SUGGESTED_PROMPTS = [
    "breast tumor",
    "liver lesion",
    "lung nodule",
    "kidney tumor",
    "brain tumor",
    "thyroid nodule",
    "pancreas lesion",
    "spleen",
    "aorta",
    "gallbladder",
    "stomach",
    "colon polyp",
    "prostate lesion",
    "skin lesion",
]

# ---------------------------------------------------------------------------
# Lazy model reference
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is None:
        from inference import get_model

        _model = get_model()
    return _model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    m = _get_model()
    return {
        "status": "ok",
        "model_loaded": m._model is not None,
        "device": m.device,
    }


@app.get("/prompts")
async def prompts():
    return {"prompts": SUGGESTED_PROMPTS}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    prompt: str = Form("breast tumor"),
    threshold: float = Form(0.1),
):
    """Accept an image + text prompt, return segmentation results."""
    try:
        contents = await image.read()
        pil_img = Image.open(__import__("io").BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)

        model = _get_model()
        result = model.predict(img_np, text_prompt=prompt, threshold=threshold)

        return JSONResponse(
            {
                "ok": True,
                "prompt": prompt,
                "threshold": threshold,
                "num_detections": len(result["scores"]),
                "scores": result["scores"],
                "boxes": result["boxes"],
                "overlay_b64": result["overlay_b64"],
                "elapsed_ms": result["elapsed_ms"],
                "image_size": [img_np.shape[1], img_np.shape[0]],
            }
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


# Serve frontend
@app.get("/")
async def index():
    return FileResponse(_FRONTEND / "index.html")


# Static files (if any CSS/JS is added later)
if _FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ---------------------------------------------------------------------------
# Run with `python backend/app.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # Add backend dir to path so `from inference import …` works
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    port = int(os.environ.get("MEDSAM3_PORT", 8000))
    logger.info(f"Starting Medical SAM3 demo on http://localhost:{port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
