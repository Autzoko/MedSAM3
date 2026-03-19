#!/usr/bin/env python3
"""
Evaluate Medical SAM3 on BIrads WebDataset shards (for HPC).

Reads .tar shards, groups slices by 3D case, runs inference on positive
slices, compares best prediction to GT max bbox.

Usage:
    python evaluate_webdataset.py --shards shards/
    python evaluate_webdataset.py --shards shards/ --max-cases 5 --prompt "lesion"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from inference import MedSAM3, _install_stubs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("eval_wds")


# ---------------------------------------------------------------------------
# Data loading from WebDataset shards
# ---------------------------------------------------------------------------

def parse_label(txt: str) -> list[list[float]]:
    """Parse label text → list of [x1, y1, x2, y2]."""
    boxes = []
    for line in txt.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if cls_id == 0 or (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
            continue
        boxes.append([x1, y1, x2, y2])
    return boxes


def load_shards(shard_dir: Path) -> dict[str, list[dict]]:
    """Load all samples from .tar shards, grouped by case_id.

    Returns: {case_id: [{"slice_idx", "image_bytes", "gt_boxes"}, ...]}
    """
    cases: dict[str, list[dict]] = defaultdict(list)

    tar_files = sorted(shard_dir.glob("birads-*.tar"))
    log.info(f"Loading {len(tar_files)} shard(s) from {shard_dir}")

    for tar_path in tar_files:
        # Read all members grouped by key prefix
        samples: dict[str, dict] = {}
        with tarfile.open(str(tar_path), "r") as tar:
            for member in tar:
                if member.isdir():
                    continue
                # Key format: case_id/slice_idx.ext
                name = member.name
                key, ext = name.rsplit(".", 1)
                if key not in samples:
                    samples[key] = {"__key__": key}
                data = tar.extractfile(member)
                if data is not None:
                    samples[key][ext] = data.read()

        for key, sample in samples.items():
            parts = key.split("/")
            if len(parts) != 2:
                continue
            case_id, slice_idx = parts

            gt_boxes = []
            if "txt" in sample:
                gt_boxes = parse_label(sample["txt"].decode("utf-8"))

            cases[case_id].append({
                "slice_idx": slice_idx,
                "image_bytes": sample.get("png", b""),
                "gt_boxes": gt_boxes,
            })

    # Sort slices within each case
    for case_id in cases:
        cases[case_id].sort(key=lambda s: s["slice_idx"])

    log.info(f"Loaded {len(cases)} cases, "
             f"{sum(len(v) for v in cases.values())} total slices")
    return dict(cases)


# ---------------------------------------------------------------------------
# Metrics (same as evaluate_birads.py)
# ---------------------------------------------------------------------------

def compute_iou(a: list[float], b: list[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def compute_giou(a: list[float], b: list[float]) -> float:
    iou = compute_iou(a, b)
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
    area_c = (ex2 - ex1) * (ey2 - ey1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
            max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = aa + ab - inter
    return iou - (area_c - union) / area_c if area_c > 0 else iou


def center_distance(a: list[float], b: list[float]) -> float:
    ca = ((a[0]+a[2])/2, (a[1]+a[3])/2)
    cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    return ((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2) ** 0.5


def box_area(b: list[float]) -> float:
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    gt_bbox: list[float]
    gt_slice_idx: str
    gt_area: float
    pred_bbox: Optional[list[float]] = None
    pred_score: float = 0.0
    pred_slice_idx: str = ""
    iou: float = 0.0
    giou: float = 0.0
    center_dist: float = float("inf")
    area_ratio: float = 0.0
    detected: bool = False
    n_slices_evaluated: int = 0
    elapsed_s: float = 0.0


def evaluate_case(
    model: MedSAM3,
    case_id: str,
    slices: list[dict],
    prompt: str,
    threshold: float,
    save_viz_dir: Optional[Path] = None,
) -> Optional[CaseResult]:
    """Evaluate one 3D case."""

    # Find GT max bbox
    best_gt = None
    for sl in slices:
        for box in sl["gt_boxes"]:
            a = (box[2]-box[0]) * (box[3]-box[1])
            if best_gt is None or a > best_gt[1]:
                best_gt = (box, a, sl["slice_idx"])

    if best_gt is None:
        return None

    gt_box, gt_area, gt_slice = best_gt
    result = CaseResult(
        case_id=case_id, gt_bbox=gt_box,
        gt_slice_idx=gt_slice, gt_area=gt_area,
    )

    positive_slices = [s for s in slices if s["gt_boxes"]]
    best_score = -1.0
    best_pred_box = None
    best_pred_slice = ""
    best_overlay_b64 = None

    t0 = time.time()
    n_eval = 0

    for sl in positive_slices:
        if not sl["image_bytes"]:
            continue
        try:
            pil = Image.open(io.BytesIO(sl["image_bytes"])).convert("RGB")
            img_np = np.array(pil)
            pred = model.predict(img_np, text_prompt=prompt, threshold=threshold)
            n_eval += 1

            if pred["scores"]:
                idx = int(np.argmax(pred["scores"]))
                score = pred["scores"][idx]
                pred_box = pred["boxes"][idx]
                if score > best_score:
                    best_score = score
                    best_pred_box = pred_box
                    best_pred_slice = sl["slice_idx"]
                    best_overlay_b64 = pred["overlay_b64"]
        except Exception as e:
            log.warning(f"  {case_id}/slice_{sl['slice_idx']}: {e}")
            continue

    result.elapsed_s = time.time() - t0
    result.n_slices_evaluated = n_eval

    if best_pred_box is not None:
        result.detected = True
        result.pred_bbox = best_pred_box
        result.pred_score = best_score
        result.pred_slice_idx = best_pred_slice
        result.iou = compute_iou(gt_box, best_pred_box)
        result.giou = compute_giou(gt_box, best_pred_box)
        result.center_dist = center_distance(gt_box, best_pred_box)
        result.area_ratio = box_area(best_pred_box) / gt_area if gt_area > 0 else 0

        # Save visualization
        if save_viz_dir and best_overlay_b64:
            import base64
            viz_path = save_viz_dir / f"{case_id}_best.png"
            viz_path.write_bytes(base64.b64decode(best_overlay_b64))

    return result


def print_summary(results: list[CaseResult], prompt: str, threshold: float) -> str:
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
    p("  MEDICAL SAM3 — BIrads Evaluation Summary (WebDataset)")
    p("=" * 78)
    p()
    p(f"  Prompt:               \"{prompt}\"")
    p(f"  Threshold:            {threshold}")
    p(f"  Total 3D cases:       {n}")
    p(f"  Cases with detection: {n_det} / {n}  ({n_det/n*100:.1f}%)")
    p()

    p("─" * 78)
    p("  DETECTION METRICS")
    p("─" * 78)
    for t in [0.1, 0.25, 0.5, 0.75]:
        hits = sum(1 for iou in ious if iou >= t)
        p(f"  Det Rate @ IoU≥{t:.2f}:  {hits}/{n}  ({hits/n*100:.1f}%)")
    p()

    if detected:
        p("─" * 78)
        p("  BOX QUALITY METRICS  (detected cases only)")
        p("─" * 78)
        def stats(vals, name, fmt=".3f"):
            a = np.array(vals)
            p(f"  {name:<28s}  mean={a.mean():{fmt}}  med={np.median(a):{fmt}}  "
              f"std={a.std():{fmt}}  min={a.min():{fmt}}  max={a.max():{fmt}}")
        stats(ious, "IoU")
        stats(gious, "GIoU")
        stats(dists, "Center Dist (px)", ".1f")
        stats(area_ratios, "Area Ratio (pred/gt)")
        stats(scores, "Confidence")
        p()

    p("─" * 78)
    p("  TIMING")
    p("─" * 78)
    total_t = sum(r.elapsed_s for r in results)
    total_sl = sum(r.n_slices_evaluated for r in results)
    p(f"  Total time:        {total_t:.0f}s  ({total_t/3600:.1f}h)")
    p(f"  Slices evaluated:  {total_sl}")
    if total_sl:
        p(f"  Avg per slice:     {total_t/total_sl:.2f}s")
    p()

    p("─" * 78)
    p("  PER-CASE RESULTS")
    p("─" * 78)
    hdr = f"  {'Case':<16s} {'Det':>3s} {'Score':>6s} {'IoU':>6s} {'GIoU':>6s} {'Dist':>7s} {'AreaR':>6s} {'#Sl':>4s} {'Time':>6s}"
    p(hdr)
    p("  " + "─" * (len(hdr) - 2))
    for r in sorted(results, key=lambda x: x.iou, reverse=True):
        if r.detected:
            p(f"  {r.case_id:<16s} {'Y':>3s} {r.pred_score:>6.3f} {r.iou:>6.3f} "
              f"{r.giou:>6.3f} {r.center_dist:>7.1f} {r.area_ratio:>6.2f} "
              f"{r.n_slices_evaluated:>4d} {r.elapsed_s:>5.0f}s")
        else:
            p(f"  {r.case_id:<16s} {'N':>3s} {'—':>6s} {'—':>6s} "
              f"{'—':>6s} {'—':>7s} {'—':>6s} "
              f"{r.n_slices_evaluated:>4d} {r.elapsed_s:>5.0f}s")

    p()
    p("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=Path("shards"))
    parser.add_argument("--prompt", type=str, default="breast tumor")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--output", type=str, default="eval_results_wds.json")
    parser.add_argument("--viz-dir", type=Path, default=None,
                        help="Save best overlay PNGs here")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pt")
    args = parser.parse_args()

    # Load data from shards
    cases = load_shards(args.shards)
    case_ids = sorted(cases.keys())
    if args.max_cases > 0:
        case_ids = case_ids[:args.max_cases]

    log.info(f"Evaluating {len(case_ids)} cases, prompt=\"{args.prompt}\"")

    # Viz dir
    viz_dir = args.viz_dir
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    _install_stubs()
    ckpt = args.checkpoint or str(ROOT / "checkpoints" / "checkpoint.pt")
    model = MedSAM3(checkpoint_path=ckpt)
    model.load()

    # Evaluate
    results: list[CaseResult] = []
    for i, cid in enumerate(case_ids):
        log.info(f"[{i+1}/{len(case_ids)}] {cid} ({len(cases[cid])} slices)...")
        r = evaluate_case(model, cid, cases[cid], args.prompt, args.threshold, viz_dir)
        if r:
            results.append(r)
            s = f"IoU={r.iou:.3f} score={r.pred_score:.3f}" if r.detected else "no det"
            log.info(f"  → {s}  ({r.n_slices_evaluated} slices, {r.elapsed_s:.0f}s)")

    # Summary
    summary = print_summary(results, args.prompt, args.threshold)

    # Save JSON
    out_path = Path(args.output)
    json_out = {
        "config": {"prompt": args.prompt, "threshold": args.threshold, "n_cases": len(results)},
        "summary": {
            "n_cases": len(results),
            "n_detected": sum(1 for r in results if r.detected),
            "detection_rate": sum(1 for r in results if r.detected) / len(results) if results else 0,
            "mean_iou": float(np.mean([r.iou for r in results if r.detected])) if any(r.detected for r in results) else 0,
            "det_rate_iou50": sum(1 for r in results if r.iou >= 0.5) / len(results) if results else 0,
        },
        "per_case": [
            {
                "case_id": r.case_id, "detected": r.detected,
                "gt_bbox": r.gt_bbox, "pred_bbox": r.pred_bbox,
                "pred_score": r.pred_score, "iou": r.iou, "giou": r.giou,
                "center_dist": r.center_dist, "area_ratio": r.area_ratio,
                "n_slices": r.n_slices_evaluated, "elapsed_s": r.elapsed_s,
            }
            for r in results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)

    # Save summary text
    Path(args.output.replace(".json", "_summary.txt")).write_text(summary, encoding="utf-8")
    log.info(f"Results: {out_path}")


if __name__ == "__main__":
    main()
