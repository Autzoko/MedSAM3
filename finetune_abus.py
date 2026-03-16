#!/usr/bin/env python3
"""Finetune Medical SAM3 on ABUS breast ultrasound dataset.

Loads pretrained SAM3 checkpoint, finetunes on ABUS 2D slices packed as
WebDataset .tar shards. Matches the Medical SAM3 paper methodology:
  - Full model finetuning (detection + segmentation)
  - Inverse-square-root LR schedule with warmup
  - LLRD γ=0.85 on vision backbone
  - BinaryOneToManyMatcher (DAC-DETR) for O2M matching
  - L_find (L1 + GIoU + focal-CE + presence) + L_seg (focal-mask + dice)

Usage:
    python finetune_abus.py --shards shards_abus/ --checkpoint checkpoints/checkpoint.pt
    python finetune_abus.py --shards shards_abus/ --checkpoint checkpoints/checkpoint.pt --epochs 20
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random
import sys
import tarfile
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup — must happen before SAM3 imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "sam3"))
sys.path.insert(0, str(ROOT / "third_party" / "Medical-SAM3"))
sys.path.insert(0, str(ROOT / "third_party" / "Medical-SAM3" / "inference"))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Stub out unavailable modules before SAM3 import
import types
for _mod in ("decord",):
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.__path__, _stub.__all__ = [], []
        _stub.__file__ = _stub.__loader__ = _stub.__spec__ = None
        _sentinel = type("_Stub", (), {"__call__": lambda self, *a, **k: None})
        _stub.cpu, _stub.VideoReader = _sentinel(), _sentinel()
        sys.modules[_mod] = _stub

# Now import SAM3 components
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.optim.schedulers import InverseSquareRootParamScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("finetune")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ABUSShardDataset(Dataset):
    """Loads ABUS data from WebDataset .tar shards."""

    def __init__(
        self,
        shard_dir: Path,
        split: str = "train",
        resolution: int = 1008,
        augment: bool = True,
    ):
        self.resolution = resolution
        self.augment = augment
        self.samples: list[dict] = []

        tar_files = sorted(shard_dir.glob(f"abus-{split}-*.tar"))
        if not tar_files:
            log.warning(f"No shards found for split '{split}' in {shard_dir}")
            return

        for tar_path in tar_files:
            members: dict[str, dict] = {}
            with tarfile.open(str(tar_path), "r") as tar:
                for member in tar:
                    if member.isdir():
                        continue
                    name = member.name
                    # Split key.ext — key is everything before first dot
                    dot = name.find(".")
                    if dot < 0:
                        continue
                    key, ext = name[:dot], name[dot + 1 :]
                    if key not in members:
                        members[key] = {}
                    data = tar.extractfile(member)
                    if data is not None:
                        members[key][ext] = data.read()

            for key, data in members.items():
                if "png" not in data or "json" not in data:
                    continue
                meta = json.loads(data["json"].decode("utf-8"))
                self.samples.append(
                    {
                        "image_bytes": data["png"],
                        "mask_bytes": data.get("mask.png", b""),  # segmentation mask
                        "bbox": meta["bbox"],   # [x1, y1, x2, y2] pixels
                        "label": meta["label"],
                        "case_id": meta["case_id"],
                        "size": meta["size"],    # [H, W]
                    }
                )

        log.info(f"Loaded {len(self.samples)} samples for split='{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Load grayscale → RGB
        img = Image.open(io.BytesIO(s["image_bytes"])).convert("RGB")
        orig_w, orig_h = img.size

        # Load segmentation mask (if available)
        mask_bytes = s.get("mask_bytes", b"")
        if mask_bytes:
            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        else:
            mask = None

        x1, y1, x2, y2 = [float(v) for v in s["bbox"]]

        # --- Augmentation ---
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask is not None:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                x1, x2 = orig_w - x2, orig_w - x1

            # Random brightness / contrast
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = ImageEnhance.Brightness(img).enhance(factor)
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = ImageEnhance.Contrast(img).enhance(factor)

        # Resize to model resolution
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)

        # Scale bbox
        sx = self.resolution / orig_w
        sy = self.resolution / orig_h
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

        # Clamp
        x1 = max(0.0, min(x1, self.resolution))
        y1 = max(0.0, min(y1, self.resolution))
        x2 = max(0.0, min(x2, self.resolution))
        y2 = max(0.0, min(y2, self.resolution))

        # Convert to normalised CxCyWH
        cx = (x1 + x2) / 2 / self.resolution
        cy = (y1 + y2) / 2 / self.resolution
        w = (x2 - x1) / self.resolution
        h = (y2 - y1) / self.resolution

        # To tensor & normalise (mean=0.5, std=0.5 → range [-1, 1])
        img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - 0.5) / 0.5

        # Mask → binary tensor [1, H, W]
        if mask is not None:
            mask_t = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
            mask_t = (mask_t > 0.5).float()
        else:
            mask_t = torch.zeros(1, self.resolution, self.resolution)

        bbox_t = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
        label = 1 if s["label"] == "malignant" else 0

        return img_t, bbox_t, label, mask_t


# ---------------------------------------------------------------------------
# Collation → BatchedDatapoint
# ---------------------------------------------------------------------------
def collate_abus(batch, text_prompt: str = "breast lesion"):
    """Collate list of (img, bbox, label, mask) into a BatchedDatapoint."""
    images, boxes_list, labels, masks_list = zip(*batch)
    B = len(images)

    img_batch = torch.stack(images)  # [B, 3, 1008, 1008]

    # --- FindStage (text-only prompting, no geometric input) ---
    find_input = FindStage(
        img_ids=torch.arange(B, dtype=torch.long),
        text_ids=torch.zeros(B, dtype=torch.long),
        input_boxes=torch.zeros(B, 1, 4, dtype=torch.float),
        input_boxes_mask=torch.ones(B, 1, dtype=torch.bool),   # all masked
        input_boxes_label=torch.zeros(B, 1, dtype=torch.long),
        input_points=torch.zeros(B, 0, 257, dtype=torch.float),
        input_points_mask=torch.zeros(B, 0, dtype=torch.bool),
        object_ids=[[] for _ in range(B)],
    )

    # --- BatchedFindTarget ---
    num_boxes_list = [b.shape[0] for b in boxes_list]
    total_boxes = sum(num_boxes_list)
    max_boxes = max(num_boxes_list) if num_boxes_list else 1

    packed_boxes = torch.cat(boxes_list, dim=0)  # [total, 4] cxcywh normalised
    padded_boxes = torch.zeros(B, max_boxes, 4)
    padded_obj_ids = torch.full((B, max_boxes), -1, dtype=torch.long)

    offset = 0
    for i, n in enumerate(num_boxes_list):
        if n > 0:
            padded_boxes[i, :n] = boxes_list[i]
            padded_obj_ids[i, :n] = torch.arange(offset, offset + n)
        offset += n

    # Pack segmentation masks: [total_boxes, H, W]
    # Each sample has 1 mask of shape [1, H, W] → squeeze and stack
    packed_masks = torch.cat(masks_list, dim=0)  # [total_boxes, H, W]
    is_valid_segment = torch.ones(total_boxes, dtype=torch.bool)

    find_target = BatchedFindTarget(
        num_boxes=torch.tensor(num_boxes_list, dtype=torch.long),
        boxes=packed_boxes,
        boxes_padded=padded_boxes,
        repeated_boxes=torch.zeros(0, 4, dtype=torch.float),
        segments=packed_masks,
        semantic_segments=None,
        is_valid_segment=is_valid_segment,
        is_exhaustive=torch.ones(B, dtype=torch.bool),
        object_ids=torch.arange(total_boxes, dtype=torch.long),
        object_ids_padded=padded_obj_ids,
    )

    # --- Dummy inference metadata (not used in training, required by dataclass) ---
    find_metadata = BatchedInferenceMetadata(
        coco_image_id=torch.arange(B, dtype=torch.long),
        original_image_id=torch.arange(B, dtype=torch.long),
        original_category_id=torch.zeros(B, dtype=torch.int),
        original_size=torch.full((B, 2), 1008, dtype=torch.long),
        object_id=torch.zeros(B, dtype=torch.long),
        frame_index=torch.zeros(B, dtype=torch.long),
        is_conditioning_only=[None] * B,
    )

    batched = BatchedDatapoint(
        img_batch=img_batch,
        find_text_batch=[text_prompt],
        find_inputs=[find_input],
        find_targets=[find_target],
        find_metadatas=[find_metadata],
    )
    return batched


# ---------------------------------------------------------------------------
# Device transfer (recursive for dataclasses)
# ---------------------------------------------------------------------------
def to_device(obj: Any, device: torch.device) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    if is_dataclass(obj) and not isinstance(obj, type):
        kwargs = {}
        for f in fields(obj):
            kwargs[f.name] = to_device(getattr(obj, f.name), device)
        return type(obj)(**kwargs)
    return obj


# ---------------------------------------------------------------------------
# LLRD (Layer-wise Learning Rate Decay) optimizer
# ---------------------------------------------------------------------------
VIT_DEPTH = 32  # SAM3 ViT depth


def _get_vit_layer_id(param_name: str) -> int:
    """Map a vision backbone parameter name to its layer index.

    Layer 0 = patch_embed (deepest), 1..32 = blocks, 33 = norm/pos_embed (shallowest).
    """
    if "patch_embed" in param_name:
        return 0
    if "blocks." in param_name:
        block_id = int(param_name.split("blocks.")[1].split(".")[0])
        return block_id + 1
    return VIT_DEPTH + 1  # norm, pos_embed, cls_token


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    lr_vision: float = 5e-5,
    lr_language: float = 5e-5,
    lr_geometry: float = 1e-4,
    weight_decay: float = 0.1,
    lrd: float = 0.85,
) -> torch.optim.AdamW:
    """Build AdamW with layer-wise LR decay on the vision backbone.

    Per the Medical SAM3 paper:
      - Vision backbone: 5e-5 base with LLRD γ=0.85
      - Transformer decoder: 3e-4
      - Language backbone: 5e-5
      - Geometry encoder: 1e-4
    """
    num_layers = VIT_DEPTH + 1  # 33 entries: 0=patch_embed, 1-32=blocks, 32=output
    layer_scales = [lrd ** (num_layers - i) for i in range(num_layers + 1)]

    param_groups: list[dict] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))

        # No weight decay on bias, LayerNorm, embeddings
        no_wd = ("bias" in name or "norm" in name.lower() or "ln" in name.lower()
                 or "embed" in name.lower())
        wd = 0.0 if no_wd else weight_decay

        if "backbone.vision_backbone.trunk" in name:
            layer_id = _get_vit_layer_id(name.split("backbone.vision_backbone.trunk.")[-1])
            scale = layer_scales[min(layer_id, len(layer_scales) - 1)]
            if "pos_embed" in name:
                scale = 1.0  # no decay for position embeddings
            plr = lr_vision * scale
        elif "backbone.language_backbone" in name or "backbone.text" in name:
            plr = lr_language
        elif "geometry_encoder" in name or "geometric_kernel" in name:
            plr = lr_geometry
        else:
            plr = lr

        param_groups.append({"params": [param], "lr": plr, "weight_decay": wd})

    optimizer = torch.optim.AdamW(param_groups)
    log.info(f"Optimizer: {len(param_groups)} param groups, "
             f"lr={lr}, lr_vis={lr_vision}, lr_lang={lr_language}, "
             f"lr_geom={lr_geometry}, lrd={lrd}, wd={weight_decay}")
    return optimizer


# ---------------------------------------------------------------------------
# Inverse-square-root LR schedule (per Medical SAM3 paper / SAM3 codebase)
# ---------------------------------------------------------------------------
def get_inverse_sqrt_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    cooldown_steps: int = 0,
    timescale: int = 10000,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Wraps SAM3's InverseSquareRootParamScheduler as a LambdaLR.

    Each param group keeps its own base_lr (set by build_optimizer).
    The scheduler returns a multiplier relative to that base_lr.
    """
    # We use base_lr=1.0 so __call__ returns the *multiplier*
    _sched = InverseSquareRootParamScheduler(
        base_lr=1.0,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        timescale=timescale,
    )

    def lr_lambda(step: int) -> float:
        where = step / max(total_steps, 1)
        return _sched(step, where)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Build loss function
# ---------------------------------------------------------------------------
def build_loss(device: torch.device, enable_segmentation: bool = True) -> Sam3LossWrapper:
    """Create loss matching Medical SAM3 paper (Table 2).

    L_find: L1(λ=5) + GIoU(λ=2) + focal-CE(λ=20) + presence(λ=20)
    L_seg:  focal-mask(λ=20) + dice(λ=1) + presence(λ=30)   [paper: λ_d=20, λ_p=1, λ_dl=30]
    O2M:    BinaryOneToManyMatcher(topk=4, α=0.3, threshold=0.4)
    """
    box_loss = Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0})
    ce_loss = IABCEMdetr(
        pos_weight=10.0,
        weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
        alpha=0.25,
        gamma=2,
        use_presence=True,
    )

    loss_fns = [box_loss, ce_loss]

    if enable_segmentation:
        mask_loss = Masks(
            weight_dict={"loss_mask": 20.0, "loss_dice": 1.0},
            focal_alpha=0.6,
            focal_gamma=2.0,
        )
        loss_fns.append(mask_loss)

    # O2M matcher: BinaryOneToManyMatcher per paper (DAC-DETR style, topk=4)
    o2m_matcher = BinaryOneToManyMatcher(
        alpha=0.3,
        threshold=0.4,
        topk=4,
    )
    loss_fn = Sam3LossWrapper(
        loss_fns_find=loss_fns,
        normalization="local",  # single-GPU
        o2m_matcher=o2m_matcher,
        o2m_weight=2.0,
    )
    loss_fn = loss_fn.to(device)
    return loss_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_clip: float = 0.1,
    log_interval: int = 10,
    text_prompt: str = "breast lesion",
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(dataloader):
        batched_dp = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(batched_dp)
            targets = [model.back_convert(t) for t in batched_dp.find_targets]
            losses = loss_fn(outputs, targets)
            core_loss = losses["core_loss"]

        if torch.isnan(core_loss) or torch.isinf(core_loss):
            log.warning(f"  Step {i}: NaN/Inf loss, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(core_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += core_loss.item()
        n_batches += 1

        if i % log_interval == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            log.info(
                f"  Epoch {epoch} [{i}/{len(dataloader)}]  "
                f"loss={core_loss.item():.4f}  lr={cur_lr:.2e}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    text_prompt: str = "breast lesion",
) -> dict:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0
    n_detected = 0
    n_total = 0

    for batch in dataloader:
        batched_dp = to_device(batch, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(batched_dp)
            targets = [model.back_convert(t) for t in batched_dp.find_targets]
            losses = loss_fn(outputs, targets)

        total_loss += losses["core_loss"].item()
        n_batches += 1

        # Quick IoU check: best predicted box vs GT for each image
        # outputs is SAM3Output; get the last step of the first (only) stage
        stage_out = list(outputs)[0]  # list of steps for stage 0
        if isinstance(stage_out, list):
            step_out = stage_out[-1]  # last step
        else:
            step_out = stage_out

        pred_boxes_xyxy = step_out.get("pred_boxes_xyxy")  # [B, Q, 4]
        pred_logits = step_out.get("pred_logits")           # [B, Q, 1]

        if pred_boxes_xyxy is not None and pred_logits is not None:
            B = pred_logits.shape[0]
            gt_boxes = batched_dp.find_targets[0].boxes  # packed cxcywh

            offset = 0
            for b in range(B):
                n_gt = batched_dp.find_targets[0].num_boxes[b].item()
                if n_gt == 0:
                    offset += n_gt
                    continue

                # Best prediction (highest logit)
                best_q = pred_logits[b, :, 0].argmax().item()
                pred = pred_boxes_xyxy[b, best_q].cpu()  # xyxy normalised

                # GT: convert cxcywh → xyxy
                gt = gt_boxes[offset].cpu()
                gt_xyxy = torch.tensor([
                    gt[0] - gt[2] / 2, gt[1] - gt[3] / 2,
                    gt[0] + gt[2] / 2, gt[1] + gt[3] / 2,
                ])

                iou = _compute_iou(pred.tolist(), gt_xyxy.tolist())
                total_iou += iou
                n_total += 1
                if iou > 0.1:
                    n_detected += 1

                offset += n_gt

    avg_loss = total_loss / max(n_batches, 1)
    avg_iou = total_iou / max(n_total, 1)
    det_rate = n_detected / max(n_total, 1)

    return {"val_loss": avg_loss, "val_iou": avg_iou, "det_rate": det_rate, "n_total": n_total}


def _compute_iou(a: list[float], b: list[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_iou: float,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model state with "detector." prefix for compatibility with SAM3's
    # checkpoint loader (used by inference.py and evaluate_webdataset.py)
    state_dict = model.state_dict()
    compat_state_dict = {f"detector.{k}": v for k, v in state_dict.items()}

    torch.save(
        {
            "model": compat_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
        },
        path,
    )
    log.info(f"Saved checkpoint → {path}")


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> tuple[int, float]:
    """Load a resume checkpoint. Returns (start_epoch, best_iou)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    # Strip "detector." prefix if present (our save format)
    stripped = {
        k.replace("detector.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(stripped, strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_iou = ckpt.get("best_iou", 0.0)
    log.info(f"Resumed from {path}, epoch={start_epoch}, best_iou={best_iou:.4f}")
    return start_epoch, best_iou


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Finetune Medical SAM3 on ABUS")
    # Data
    parser.add_argument("--shards", type=Path, default=Path("shards_abus"))
    parser.add_argument("--text-prompt", type=str, default="breast lesion")
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained SAM3 checkpoint path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from this checkpoint")
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="LR for transformer decoder (paper: 3e-4)")
    parser.add_argument("--lr-vision", type=float, default=5e-5,
                        help="Base LR for vision backbone before LLRD (paper: 5e-5)")
    parser.add_argument("--lr-language", type=float, default=5e-5,
                        help="LR for language backbone (paper: 5e-5)")
    parser.add_argument("--lr-geometry", type=float, default=1e-4,
                        help="LR for geometry encoder (paper: 1e-4)")
    parser.add_argument("--lrd", type=float, default=0.85,
                        help="Layer-wise learning rate decay for vision backbone (paper: 0.85)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--val-freq", type=int, default=2,
                        help="Validate every N epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("finetune_output"))
    parser.add_argument("--log-interval", type=int, default=10)
    # Device
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log.info("Using CPU (finetuning will be very slow)")
    use_amp = torch.cuda.is_available() and not args.no_amp

    # --- Data ---
    log.info("Loading training data from shards...")
    train_ds = ABUSShardDataset(args.shards, split="train", augment=True)
    val_ds = ABUSShardDataset(args.shards, split="val", augment=False)

    collate_fn = lambda batch: collate_abus(batch, text_prompt=args.text_prompt)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # --- Model ---
    log.info("Building SAM3 model...")
    ckpt_path = args.checkpoint or str(ROOT / "checkpoints" / "checkpoint.pt")
    bpe_path = None
    # Try to find BPE vocab
    for candidate in [
        ROOT / "third_party" / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        ROOT / "third_party" / "Medical-SAM3" / "inference" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]:
        if candidate.exists():
            bpe_path = str(candidate)
            break

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=False,
        checkpoint_path=ckpt_path,
        load_from_HF=False,
        enable_segmentation=True,  # paper trains with segmentation head
    )
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params / 1e6:.1f}M params, {n_trainable / 1e6:.1f}M trainable")

    # --- Loss (detection + segmentation, per paper) ---
    loss_fn = build_loss(device, enable_segmentation=True)

    # --- Optimizer ---
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        lr_vision=args.lr_vision,
        lr_language=args.lr_language,
        lr_geometry=args.lr_geometry,
        weight_decay=args.weight_decay,
        lrd=args.lrd,
    )

    # --- Scheduler (inverse-square-root, per Medical SAM3 paper) ---
    total_steps = len(train_loader) * args.epochs
    scheduler = get_inverse_sqrt_schedule(
        optimizer, args.warmup_steps, total_steps,
        cooldown_steps=0, timescale=10000,
    )

    # --- AMP scaler ---
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Resume ---
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, best_iou = load_resume_checkpoint(
                resume_path, model, optimizer, scheduler, scaler
            )

    # --- Training loop ---
    log.info(f"Training for {args.epochs} epochs, {len(train_loader)} steps/epoch")
    log.info(f"Text prompt: \"{args.text_prompt}\", AMP: {use_amp}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        avg_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, scheduler, scaler,
            device, epoch, use_amp, args.grad_clip, args.log_interval,
            args.text_prompt,
        )
        dt = time.time() - t0
        log.info(f"Epoch {epoch} done — avg_loss={avg_loss:.4f}  time={dt:.0f}s")

        # Validation
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            val_metrics = validate(
                model, loss_fn, val_loader, device, use_amp, args.text_prompt
            )
            log.info(
                f"  VAL  loss={val_metrics['val_loss']:.4f}  "
                f"IoU={val_metrics['val_iou']:.4f}  "
                f"det@0.1={val_metrics['det_rate']:.1%}  "
                f"({val_metrics['n_total']} samples)"
            )

            # Save best
            if val_metrics["val_iou"] > best_iou:
                best_iou = val_metrics["val_iou"]
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, best_iou,
                    args.output_dir / "best_checkpoint.pt",
                )

        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, best_iou,
            args.output_dir / "latest_checkpoint.pt",
        )

    log.info(f"Training complete. Best val IoU: {best_iou:.4f}")
    log.info(f"Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    main()
