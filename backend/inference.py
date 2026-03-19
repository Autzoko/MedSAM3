"""Medical SAM3 inference wrapper.

Loads the Medical-SAM3 model once, provides a high-level
``predict(image, text_prompt)`` that returns masks, bounding boxes,
confidence scores, and a base64-encoded overlay image.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure third-party repos are importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_MEDSAM3_DIR = _ROOT / "third_party" / "Medical-SAM3"
_SAM3_DIR = _ROOT / "third_party" / "sam3"

for p in (_MEDSAM3_DIR, _SAM3_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

# Also add Medical-SAM3/inference subdir if it exists
_INF_DIR = _MEDSAM3_DIR / "inference"
if _INF_DIR.is_dir() and str(_INF_DIR) not in sys.path:
    sys.path.insert(0, str(_INF_DIR))


# ---------------------------------------------------------------------------
# Stub out modules that SAM3 imports transitively but are not needed for
# inference on macOS (video decoding, triton GPU kernels, etc.).
# Must be installed AFTER torch loads (torch uses triton internally)
# but BEFORE sam3 is imported.
# ---------------------------------------------------------------------------
import types

_STUBS_INSTALLED = False

def _install_stubs() -> None:
    """Register fake modules so SAM3's transitive imports don't fail."""
    global _STUBS_INSTALLED
    if _STUBS_INSTALLED:
        return
    _STUBS_INSTALLED = True

    for module_name in ("decord",):
        if module_name not in sys.modules:
            stub = types.ModuleType(module_name)
            stub.__path__ = []
            stub.__all__ = []
            stub.__file__ = None
            stub.__loader__ = None
            stub.__spec__ = None
            # Provide callable placeholders for `from decord import cpu, VideoReader`
            _sentinel = type("_Stub", (), {"__call__": lambda self, *a, **k: None})
            stub.cpu = _sentinel()
            stub.VideoReader = _sentinel()
            sys.modules[module_name] = stub


# ---------------------------------------------------------------------------
# Overlay colour
# ---------------------------------------------------------------------------
OVERLAY_COLOUR = (0, 200, 255, 100)  # cyan-ish, semi-transparent
BORDER_COLOUR = (0, 200, 255, 255)
BOX_COLOUR = (255, 180, 0, 200)


def _choose_device() -> str:
    """Pick the best available device: cuda > cpu.

    MPS is skipped — SAM3 hits internal bugs (grid_sample, pin_memory).
    Override with MEDSAM3_DEVICE env var if needed.
    """
    import torch

    override = os.environ.get("MEDSAM3_DEVICE", "").strip().lower()
    if override:
        logger.info(f"Using device from MEDSAM3_DEVICE={override}")
        return override

    if torch.cuda.is_available():
        logger.info(f"Using CUDA ({torch.cuda.get_device_name(0)})")
        return "cuda"

    logger.info("Using CPU (no CUDA available, MPS skipped)")
    return "cpu"


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
class MedSAM3:
    """Thin wrapper around Medical-SAM3's SAM3Model."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        confidence_threshold: float = 0.1,
    ):
        self.checkpoint_path = checkpoint_path or str(
            _ROOT / "checkpoints" / "checkpoint.pt"
        )
        self.confidence_threshold = confidence_threshold
        self.device = _choose_device()
        self._model = None

    # -- lazy load --------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return

        t0 = time.time()
        logger.info("Loading Medical-SAM3 model (this takes a while on first run)…")

        # Stub out unavailable modules (decord etc.) — must happen after
        # torch is loaded (by _choose_device) but before sam3 import
        _install_stubs()

        # Find the BPE vocab file — search known locations
        _bpe_name = "bpe_simple_vocab_16e6.txt.gz"
        _bpe_candidates = [
            _SAM3_DIR / "sam3" / "assets" / _bpe_name,
            _SAM3_DIR / "assets" / _bpe_name,
            _MEDSAM3_DIR / "inference" / "sam3" / "assets" / _bpe_name,
        ]
        _bpe_path = None
        for c in _bpe_candidates:
            if c.exists():
                _bpe_path = c.resolve()
                break
        if _bpe_path is None:
            # Search recursively as last resort
            found = list(_ROOT.rglob(_bpe_name))
            if found:
                _bpe_path = found[0].resolve()
        if _bpe_path is None:
            raise FileNotFoundError(
                f"Cannot find {_bpe_name}. Searched:\n"
                + "\n".join(f"  {c}" for c in _bpe_candidates)
            )
        logger.info(f"BPE vocab: {_bpe_path}")

        # Import the upstream inference helper
        try:
            from sam3_inference import SAM3Model
        except ImportError as exc:
            raise ImportError(
                f"Cannot import sam3_inference: {exc}\n"
                "Make sure you ran install.sh and the Medical-SAM3 repo is "
                "cloned into third_party/Medical-SAM3."
            ) from exc

        # Monkey-patch SAM3_ROOT so sam3_inference.load_model finds BPE.
        # sam3_inference tries SAM3_ROOT/sam3/assets/ then SAM3_ROOT/assets/.
        # Set SAM3_ROOT so one of those resolves to the found BPE file.
        import sam3_inference as _si
        _assets_dir = _bpe_path.parent          # .../assets/
        if _assets_dir.parent.name == "sam3":
            _si.SAM3_ROOT = _assets_dir.parent.parent  # .../sam3/sam3/assets → SAM3_ROOT=.../sam3
        else:
            _si.SAM3_ROOT = _assets_dir.parent          # .../assets → SAM3_ROOT=.../ (fallback path)

        self._model = SAM3Model(
            confidence_threshold=self.confidence_threshold,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
        )
        self._model.load_model()

        dt = time.time() - t0
        logger.info(f"Model loaded in {dt:.1f}s on {self.device}")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    # -- public API -------------------------------------------------------
    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        threshold: Optional[float] = None,
    ) -> dict:
        """Run segmentation.

        Parameters
        ----------
        image : np.ndarray  (H, W, 3) uint8 RGB
        text_prompt : str    e.g. "breast tumor"
        threshold : float    override confidence threshold

        Returns
        -------
        dict with keys:
            masks        – list of binary masks (H, W) as uint8
            boxes        – list of [x_min, y_min, x_max, y_max]
            scores       – list of float confidence scores
            overlay_b64  – base64 PNG of the overlay on top of the input
            elapsed_ms   – inference time in ms
        """
        thr = threshold if threshold is not None else self.confidence_threshold
        m = self.model
        old_thr = m.confidence_threshold
        m.confidence_threshold = thr

        t0 = time.time()

        # encode
        state = m.encode_image(image)

        # predict via text prompt
        mask = m.predict_text(state, text_prompt)

        elapsed = (time.time() - t0) * 1000
        m.confidence_threshold = old_thr

        masks: list[np.ndarray] = []
        boxes: list[list[int]] = []
        scores: list[float] = []

        if mask is not None:
            # Squeeze to 2D (H, W) — upstream may return (1, H, W) or (1, 1, H, W)
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                logger.warning(f"Unexpected mask shape after squeeze: {mask.shape}")
                mask = mask.reshape(mask.shape[-2], mask.shape[-1])

            masks.append(mask)

            # Use model's own confidence score
            score = m.get_confidence(state) if hasattr(m, "get_confidence") else 0.95
            scores.append(float(score))

            # Use model-predicted boxes if available (already in pixel coords),
            # otherwise derive from mask
            if "boxes" in state and state["boxes"] is not None and len(state["boxes"]) > 0:
                import torch
                best_idx = torch.argmax(state["scores"]).item()
                model_box = state["boxes"][best_idx].cpu().numpy().tolist()
                boxes.append([int(round(c)) for c in model_box])
            else:
                bbox = _bbox_from_mask(mask)
                if bbox is not None:
                    boxes.append(list(bbox))
                else:
                    boxes.append([0, 0, mask.shape[1], mask.shape[0]])

        overlay_b64 = _make_overlay(image, masks, boxes, scores)

        return {
            "masks": [m.tolist() for m in masks],
            "boxes": boxes,
            "scores": scores,
            "overlay_b64": overlay_b64,
            "elapsed_ms": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _make_overlay(
    image: np.ndarray,
    masks: list[np.ndarray],
    boxes: list[list[int]],
    scores: list[float],
) -> str:
    """Render masks + boxes on the image, return base64-encoded PNG."""
    from PIL import ImageDraw, ImageFilter

    img_h, img_w = image.shape[:2]
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    for i, mask in enumerate(masks):
        # Resize mask to match image if needed
        if mask.shape[0] != img_h or mask.shape[1] != img_w:
            mask_pil = Image.fromarray(mask.astype(np.uint8), "L")
            mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
            mask = np.array(mask_pil)

        # Semi-transparent mask fill
        mask_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = OVERLAY_COLOUR
        mask_layer = Image.fromarray(mask_rgba, "RGBA")
        overlay = Image.alpha_composite(overlay, mask_layer)

        # Contour: dilated mask minus original mask
        m_pil = Image.fromarray((mask * 255).astype(np.uint8), "L")
        dilated = m_pil.filter(ImageFilter.MaxFilter(5))
        contour = np.array(dilated).astype(int) - np.array(m_pil).astype(int)
        contour_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        contour_rgba[contour > 0] = BORDER_COLOUR
        overlay = Image.alpha_composite(overlay, Image.fromarray(contour_rgba, "RGBA"))

    # Draw boxes and labels on the overlay
    draw = ImageDraw.Draw(overlay)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOUR[:3], width=2)
        label = f"#{i+1}  {scores[i]:.0%}" if i < len(scores) else f"#{i+1}"
        # Background for label readability
        tw = draw.textlength(label)
        draw.rectangle([x0, y0, x0 + tw + 8, y0 + 16], fill=(0, 0, 0, 180))
        draw.text((x0 + 4, y0 + 1), label, fill=(255, 255, 255, 230))

    result = Image.alpha_composite(base, overlay).convert("RGB")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Module-level singleton (created once, loaded lazily)
# ---------------------------------------------------------------------------
_instance: Optional[MedSAM3] = None


def get_model() -> MedSAM3:
    global _instance
    if _instance is None:
        _instance = MedSAM3()
    return _instance
