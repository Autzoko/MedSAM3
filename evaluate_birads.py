#!/usr/bin/env python3
"""
Evaluate Medical SAM3 on the BIrads breast ultrasound dataset.

Strategy:
  - For each 3D case folder, run inference on every positive slice.
  - The GT labels the "maximum bbox" per case; we compare the best
    predicted bbox (highest confidence) against it.
  - Metrics: IoU, GIoU, precision/recall at IoU thresholds, center
    distance, size error, detection rate.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python evaluate_birads.py
    PYTORCH_ENABLE_MPS_FALLBACK=1 python evaluate_birads.py --max-cases 5   # quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Setup paths & stubs (same as backend/inference.py)
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

# Import inference wrapper (which handles stubs, model loading, etc.)
from inference import MedSAM3, _install_stubs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/Volumes/Lang/Research/Data/3D Ultrasound/processed/BIrads")
IMAGES_DIR = DATA_ROOT / "images"
LABELS_DIR = DATA_ROOT / "labels"
PROMPT = "breast tumor"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_label_file(path: Path) -> list[list[float]]:
    """Parse a label .txt file → list of [class_id, x1, y1, x2, y2].
    Returns empty list for negative slices (class 0 or all-zero bbox)."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if cls_id == 0 or (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
            continue
        boxes.append([x1, y1, x2, y2])
    return boxes


def get_max_gt_bbox(case_id: str) -> Optional[dict]:
    """Find the GT bbox with the largest area across all slices in a case.
    Returns dict with keys: bbox [x1,y1,x2,y2], slice_idx, area, slice_path."""
    label_dir = LABELS_DIR / case_id
    image_dir = IMAGES_DIR / case_id
    if not label_dir.is_dir():
        return None

    best = None
    for lbl_file in sorted(label_dir.glob("slice_*.txt")):
        boxes = parse_label_file(lbl_file)
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if best is None or area > best["area"]:
                idx = lbl_file.stem.replace("slice_", "")
                best = {
                    "bbox": box,
                    "slice_idx": idx,
                    "area": area,
                    "slice_path": str(image_dir / f"slice_{idx}.png"),
                }
    return best


def get_positive_slices(case_id: str) -> list[dict]:
    """Get all slices with GT boxes for a case."""
    label_dir = LABELS_DIR / case_id
    image_dir = IMAGES_DIR / case_id
    slices = []
    for lbl_file in sorted(label_dir.glob("slice_*.txt")):
        boxes = parse_label_file(lbl_file)
        if boxes:
            idx = lbl_file.stem.replace("slice_", "")
            slices.append({
                "slice_idx": idx,
                "image_path": str(image_dir / f"slice_{idx}.png"),
                "gt_boxes": boxes,
            })
    return slices


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_giou(box_a: list[float], box_b: list[float]) -> float:
    """Generalized IoU (GIoU)."""
    iou = compute_iou(box_a, box_b)
    # Enclosing box
    ex1 = min(box_a[0], box_b[0])
    ey1 = min(box_a[1], box_b[1])
    ex2 = max(box_a[2], box_b[2])
    ey2 = max(box_a[3], box_b[3])
    area_c = (ex2 - ex1) * (ey2 - ey1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    inter = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * \
            max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    union = area_a + area_b - inter
    if area_c == 0:
        return iou
    return iou - (area_c - union) / area_c


def center_distance(box_a: list[float], box_b: list[float]) -> float:
    """Euclidean distance between box centers."""
    ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
    cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    return ((ca[0] - cb[0])**2 + (ca[1] - cb[1])**2) ** 0.5


def box_area(box: list[float]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    gt_bbox: list[float]
    gt_slice_idx: str
    gt_area: float
    # Best prediction across all positive slices
    pred_bbox: Optional[list[float]] = None
    pred_score: float = 0.0
    pred_slice_idx: str = ""
    # Metrics
    iou: float = 0.0
    giou: float = 0.0
    center_dist: float = float("inf")
    area_ratio: float = 0.0  # pred_area / gt_area
    detected: bool = False  # any prediction at all?
    # Timing
    n_slices_evaluated: int = 0
    elapsed_s: float = 0.0


def evaluate_case(
    model: MedSAM3,
    case_id: str,
    prompt: str = PROMPT,
    threshold: float = 0.05,
) -> Optional[CaseResult]:
    """Evaluate one 3D case: run inference on positive slices,
    pick the best prediction, compare to GT max bbox."""

    gt = get_max_gt_bbox(case_id)
    if gt is None:
        log.warning(f"  {case_id}: no GT bbox found, skipping")
        return None

    positive_slices = get_positive_slices(case_id)
    if not positive_slices:
        log.warning(f"  {case_id}: no positive slices, skipping")
        return None

    result = CaseResult(
        case_id=case_id,
        gt_bbox=gt["bbox"],
        gt_slice_idx=gt["slice_idx"],
        gt_area=gt["area"],
    )

    best_score = -1.0
    best_pred_bbox = None
    best_pred_slice = ""

    t0 = time.time()
    n_eval = 0

    for sl in positive_slices:
        img_path = sl["image_path"]
        if not Path(img_path).exists():
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img)

            pred = model.predict(img_np, text_prompt=prompt, threshold=threshold)

            n_eval += 1

            if pred["scores"]:
                # Find the best scoring prediction for this slice
                max_idx = int(np.argmax(pred["scores"]))
                score = pred["scores"][max_idx]
                pred_box = pred["boxes"][max_idx]

                if score > best_score:
                    best_score = score
                    best_pred_bbox = pred_box
                    best_pred_slice = sl["slice_idx"]

        except Exception as e:
            log.warning(f"  {case_id}/slice_{sl['slice_idx']}: inference error: {e}")
            continue

    result.elapsed_s = time.time() - t0
    result.n_slices_evaluated = n_eval

    if best_pred_bbox is not None:
        result.detected = True
        result.pred_bbox = best_pred_bbox
        result.pred_score = best_score
        result.pred_slice_idx = best_pred_slice
        result.iou = compute_iou(gt["bbox"], best_pred_bbox)
        result.giou = compute_giou(gt["bbox"], best_pred_bbox)
        result.center_dist = center_distance(gt["bbox"], best_pred_bbox)
        pred_a = box_area(best_pred_bbox)
        result.area_ratio = pred_a / gt["area"] if gt["area"] > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[CaseResult]) -> str:
    """Print and return a formatted evaluation summary."""

    n = len(results)
    detected = [r for r in results if r.detected]
    n_det = len(detected)

    ious = [r.iou for r in detected]
    gious = [r.giou for r in detected]
    dists = [r.center_dist for r in detected]
    area_ratios = [r.area_ratio for r in detected]
    scores = [r.pred_score for r in detected]

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 78)
    p("  MEDICAL SAM3 — BIrads Breast Ultrasound Evaluation Summary")
    p("=" * 78)
    p()
    p(f"  Prompt:              \"{PROMPT}\"")
    p(f"  Total 3D cases:      {n}")
    p(f"  Cases with detection: {n_det} / {n}  ({n_det/n*100:.1f}%)")
    p()

    p("─" * 78)
    p("  DETECTION METRICS")
    p("─" * 78)

    iou_thresholds = [0.1, 0.25, 0.5, 0.75]
    for t in iou_thresholds:
        hits = sum(1 for iou in ious if iou >= t)
        p(f"  Detection Rate @ IoU≥{t:.2f}:  {hits}/{n}  ({hits/n*100:.1f}%)")
    p()

    if detected:
        p("─" * 78)
        p("  BOX QUALITY METRICS  (over detected cases)")
        p("─" * 78)

        def stats(vals, name, fmt=".3f"):
            arr = np.array(vals)
            p(f"  {name:<28s}  mean={arr.mean():{fmt}}  median={np.median(arr):{fmt}}  "
              f"std={arr.std():{fmt}}  min={arr.min():{fmt}}  max={arr.max():{fmt}}")

        stats(ious, "IoU")
        stats(gious, "GIoU")
        stats(dists, "Center Distance (px)", ".1f")
        stats(area_ratios, "Area Ratio (pred/gt)")
        stats(scores, "Confidence Score")
        p()

    p("─" * 78)
    p("  TIMING")
    p("─" * 78)
    total_time = sum(r.elapsed_s for r in results)
    total_slices = sum(r.n_slices_evaluated for r in results)
    p(f"  Total inference time: {total_time:.1f}s")
    p(f"  Total slices evaluated: {total_slices}")
    if total_slices > 0:
        p(f"  Avg time per slice:  {total_time/total_slices:.2f}s")
    p(f"  Avg time per case:   {total_time/n:.1f}s")
    p()

    # Per-case table
    p("─" * 78)
    p("  PER-CASE RESULTS")
    p("─" * 78)
    header = f"  {'Case':<16s} {'Det?':>4s} {'Score':>6s} {'IoU':>6s} {'GIoU':>6s} {'CtrDist':>8s} {'AreaR':>6s} {'Slices':>6s} {'Time':>6s}"
    p(header)
    p("  " + "─" * (len(header) - 2))

    for r in sorted(results, key=lambda x: x.iou, reverse=True):
        if r.detected:
            p(f"  {r.case_id:<16s} {'Y':>4s} {r.pred_score:>6.3f} {r.iou:>6.3f} "
              f"{r.giou:>6.3f} {r.center_dist:>8.1f} {r.area_ratio:>6.2f} "
              f"{r.n_slices_evaluated:>6d} {r.elapsed_s:>5.1f}s")
        else:
            p(f"  {r.case_id:<16s} {'N':>4s} {'—':>6s} {'—':>6s} "
              f"{'—':>6s} {'—':>8s} {'—':>6s} "
              f"{r.n_slices_evaluated:>6d} {r.elapsed_s:>5.1f}s")

    p()
    p("=" * 78)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global PROMPT

    parser = argparse.ArgumentParser(description="Evaluate Medical SAM3 on BIrads")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit cases (0=all)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Text prompt")
    parser.add_argument("--output", type=str, default="eval_results_birads.json",
                        help="Output JSON path")
    args = parser.parse_args()

    PROMPT = args.prompt

    # Discover cases
    case_ids = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])
    if args.max_cases > 0:
        case_ids = case_ids[:args.max_cases]

    log.info(f"Evaluating {len(case_ids)} cases with prompt=\"{PROMPT}\"")

    # Load model
    _install_stubs()
    model = MedSAM3()
    log.info("Loading model (first call)...")
    model.load()
    log.info("Model ready.")

    # Evaluate
    results: list[CaseResult] = []
    for i, case_id in enumerate(case_ids):
        log.info(f"[{i+1}/{len(case_ids)}] Evaluating case {case_id}...")
        r = evaluate_case(model, case_id, prompt=PROMPT, threshold=args.threshold)
        if r is not None:
            results.append(r)
            status = f"IoU={r.iou:.3f}" if r.detected else "no detection"
            log.info(f"  → {status}  ({r.n_slices_evaluated} slices, {r.elapsed_s:.1f}s)")

    # Summary
    summary_text = print_summary(results)

    # Save detailed results as JSON
    out_path = ROOT / args.output
    json_results = []
    for r in results:
        json_results.append({
            "case_id": r.case_id,
            "gt_bbox": r.gt_bbox,
            "gt_slice": r.gt_slice_idx,
            "gt_area": r.gt_area,
            "pred_bbox": r.pred_bbox,
            "pred_score": r.pred_score,
            "pred_slice": r.pred_slice_idx,
            "detected": r.detected,
            "iou": r.iou,
            "giou": r.giou,
            "center_dist": r.center_dist,
            "area_ratio": r.area_ratio,
            "n_slices": r.n_slices_evaluated,
            "elapsed_s": r.elapsed_s,
        })

    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "prompt": PROMPT,
                "threshold": args.threshold,
                "n_cases": len(results),
                "data_root": str(DATA_ROOT),
            },
            "summary": {
                "n_cases": len(results),
                "n_detected": sum(1 for r in results if r.detected),
                "detection_rate": sum(1 for r in results if r.detected) / len(results) if results else 0,
                "mean_iou": float(np.mean([r.iou for r in results if r.detected])) if any(r.detected for r in results) else 0,
                "mean_giou": float(np.mean([r.giou for r in results if r.detected])) if any(r.detected for r in results) else 0,
                "det_rate_iou50": sum(1 for r in results if r.iou >= 0.5) / len(results) if results else 0,
            },
            "per_case": json_results,
        }, f, indent=2)

    log.info(f"Results saved to {out_path}")

    # Save summary text
    summary_path = ROOT / "eval_summary_birads.txt"
    summary_path.write_text(summary_text)
    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
