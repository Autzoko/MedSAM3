#!/usr/bin/env python3
"""Run Medical SAM3 inference on WebDataset shards (for HPC).

Processes ALL slices (not just GT-positive), outputs per-slice predictions
and per-case aggregated results. Compatible with shards from pack_birads_shards.py.

Usage:
    python infer_webdataset.py --shards shards_birads/ --checkpoint finetune_output_2gpu/best_checkpoint.pt
    python infer_webdataset.py --shards shards_birads/ --prompt "breast lesion" --threshold 0.1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import tarfile
import time
from collections import defaultdict
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
    force=True,
)
log = logging.getLogger("infer_wds")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_label(txt: str) -> list[list[float]]:
    """Parse label text -> list of [x1, y1, x2, y2]."""
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
        samples: dict[str, dict] = {}
        with tarfile.open(str(tar_path), "r") as tar:
            for member in tar:
                if member.isdir():
                    continue
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

    for case_id in cases:
        cases[case_id].sort(key=lambda s: s["slice_idx"])

    log.info(f"Loaded {len(cases)} cases, "
             f"{sum(len(v) for v in cases.values())} total slices")
    return dict(cases)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(a: list[float], b: list[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_case(
    model: MedSAM3,
    case_id: str,
    slices: list[dict],
    prompt: str,
    threshold: float,
    save_viz_dir: Optional[Path] = None,
) -> dict:
    """Run inference on all slices of one case.

    Returns per-case result with per-slice predictions.
    """
    per_slice = []
    best_score = -1.0
    best_pred = None

    t0 = time.time()
    n_eval = 0

    for sl in slices:
        if not sl["image_bytes"]:
            continue

        slice_result = {
            "slice_idx": sl["slice_idx"],
            "gt_boxes": sl["gt_boxes"],
            "pred_boxes": [],
            "pred_scores": [],
        }

        try:
            pil = Image.open(io.BytesIO(sl["image_bytes"])).convert("RGB")
            img_np = np.array(pil)
            pred = model.predict(img_np, text_prompt=prompt, threshold=threshold)
            n_eval += 1

            if pred["scores"]:
                slice_result["pred_boxes"] = pred["boxes"]
                slice_result["pred_scores"] = [float(s) for s in pred["scores"]]

                # Track best prediction across all slices
                idx = int(np.argmax(pred["scores"]))
                if pred["scores"][idx] > best_score:
                    best_score = pred["scores"][idx]
                    best_pred = {
                        "box": pred["boxes"][idx],
                        "score": float(pred["scores"][idx]),
                        "slice_idx": sl["slice_idx"],
                        "overlay_b64": pred.get("overlay_b64"),
                    }

        except Exception as e:
            log.warning(f"  {case_id}/{sl['slice_idx']}: {e}")
            slice_result["error"] = str(e)

        per_slice.append(slice_result)

    elapsed = time.time() - t0

    # Evaluate against GT if available
    gt_boxes_all = []
    for sl in slices:
        for box in sl["gt_boxes"]:
            gt_boxes_all.append((box, sl["slice_idx"]))

    # Find largest GT box
    best_gt = None
    if gt_boxes_all:
        for box, sidx in gt_boxes_all:
            area = (box[2] - box[0]) * (box[3] - box[1])
            if best_gt is None or area > best_gt[1]:
                best_gt = (box, area, sidx)

    iou = 0.0
    if best_gt and best_pred:
        iou = compute_iou(best_gt[0], best_pred["box"])

    # Save visualization of best prediction
    if save_viz_dir and best_pred and best_pred.get("overlay_b64"):
        save_viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = save_viz_dir / f"{case_id}_best.png"
        viz_path.write_bytes(base64.b64decode(best_pred["overlay_b64"]))

    result = {
        "case_id": case_id,
        "n_slices": len(slices),
        "n_evaluated": n_eval,
        "elapsed_s": round(elapsed, 2),
        "best_prediction": {
            "box": best_pred["box"] if best_pred else None,
            "score": best_pred["score"] if best_pred else 0.0,
            "slice_idx": best_pred["slice_idx"] if best_pred else None,
        },
        "gt_bbox": best_gt[0] if best_gt else None,
        "iou": round(iou, 4),
        "per_slice": per_slice,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run Medical SAM3 inference on WebDataset shards"
    )
    parser.add_argument("--shards", type=Path, required=True,
                        help="Directory containing birads-*.tar shards")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to finetuned checkpoint (default: checkpoints/checkpoint.pt)")
    parser.add_argument("--prompt", type=str, default="breast lesion",
                        help="Text prompt for detection")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Confidence threshold")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Limit number of cases (0 = all)")
    parser.add_argument("--output", type=Path, default=Path("infer_results.json"),
                        help="Output JSON file")
    parser.add_argument("--viz-dir", type=Path, default=None,
                        help="Directory to save best prediction visualizations")
    parser.add_argument("--skip-negative", action="store_true",
                        help="Only run inference on slices with GT annotations")
    args = parser.parse_args()

    _install_stubs()

    # Load data
    cases = load_shards(args.shards)
    case_ids = sorted(cases.keys())
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]

    # Load model
    log.info(f"Loading model (checkpoint: {args.checkpoint or 'default'})...")
    model = MedSAM3(
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.threshold,
    )
    model.load()

    # Run inference
    all_results = []
    total_t0 = time.time()

    for i, case_id in enumerate(case_ids):
        slices = cases[case_id]
        if args.skip_negative:
            slices = [s for s in slices if s["gt_boxes"]]

        log.info(f"[{i+1}/{len(case_ids)}] {case_id}: {len(slices)} slices")

        result = infer_case(
            model, case_id, slices, args.prompt, args.threshold, args.viz_dir
        )
        all_results.append(result)

        # Progress
        has_gt = result["gt_bbox"] is not None
        det = result["best_prediction"]["score"] > 0
        log.info(
            f"  → score={result['best_prediction']['score']:.3f}  "
            f"iou={result['iou']:.3f}  "
            f"{'GT' if has_gt else 'no-GT'}  "
            f"({result['elapsed_s']:.1f}s)"
        )

    total_elapsed = time.time() - total_t0

    # Summary
    cases_with_gt = [r for r in all_results if r["gt_bbox"] is not None]
    detected = [r for r in cases_with_gt if r["best_prediction"]["score"] > 0]
    ious = [r["iou"] for r in detected]

    summary = {
        "n_cases": len(all_results),
        "n_cases_with_gt": len(cases_with_gt),
        "n_detected": len(detected),
        "detection_rate": len(detected) / max(len(cases_with_gt), 1),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "det_rate_iou50": sum(1 for x in ious if x >= 0.5) / max(len(cases_with_gt), 1),
        "total_elapsed_s": round(total_elapsed, 1),
    }

    log.info(f"\n=== Summary ===")
    log.info(f"Cases: {summary['n_cases']} ({summary['n_cases_with_gt']} with GT)")
    log.info(f"Detected: {summary['n_detected']}/{summary['n_cases_with_gt']} "
             f"({summary['detection_rate']:.1%})")
    log.info(f"Mean IoU: {summary['mean_iou']:.4f}")
    log.info(f"Det@IoU≥0.5: {summary['det_rate_iou50']:.1%}")
    log.info(f"Total time: {summary['total_elapsed_s']}s")

    # Save results (strip per_slice for compact output, keep in full output)
    output = {
        "config": {
            "prompt": args.prompt,
            "threshold": args.threshold,
            "checkpoint": args.checkpoint or "default",
        },
        "summary": summary,
        "per_case": all_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {args.output}")

    # Also save compact summary (without per-slice details)
    summary_path = args.output.with_suffix(".summary.json")
    compact = {
        "config": output["config"],
        "summary": summary,
        "per_case": [
            {k: v for k, v in r.items() if k != "per_slice"}
            for r in all_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    log.info(f"Compact summary saved to {summary_path}")


if __name__ == "__main__":
    main()
