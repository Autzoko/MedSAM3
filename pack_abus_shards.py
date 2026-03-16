#!/usr/bin/env python3
"""Pack ABUS 2D dataset into WebDataset .tar shards for HPC transfer.

Reads ABUS_2D directory with Train/Validation/Test splits, derives bounding
boxes from binary masks, and packs images + metadata into .tar shards.

Usage:
    python pack_abus_shards.py --data-dir /Volumes/Autzoko/ABUS_2D --out-dir shards_abus
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def derive_bbox(mask_path: Path) -> list[int] | None:
    """Derive [x1, y1, x2, y2] bounding box from a binary mask."""
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def pack_split(
    data_dir: Path,
    csv_path: Path,
    out_dir: Path,
    split: str,
    samples_per_shard: int = 500,
) -> None:
    """Pack one data split into .tar shards."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    print(f"[{split}] {len(rows)} samples from {csv_path.name}")

    shard_idx = 0
    sample_count = 0
    tar = None

    for row in rows:
        img_path = data_dir / row["image_file"]
        mask_path = data_dir / row["mask_file"]

        if not img_path.exists():
            print(f"  SKIP (no image): {row['image_file']}")
            continue

        bbox = derive_bbox(mask_path) if mask_path.exists() else None
        if bbox is None:
            print(f"  SKIP (empty mask): {row['mask_file']}")
            continue

        # Open new shard if needed
        if sample_count % samples_per_shard == 0:
            if tar:
                tar.close()
            tar_path = out_dir / f"abus-{split}-{shard_idx:06d}.tar"
            tar = tarfile.open(str(tar_path), "w")
            shard_idx += 1

        # Build key
        label = row["label"]
        case_id = int(row["case_id"])
        slice_idx = int(row["slice_idx"])
        key = f"{label}_{case_id:03d}_{slice_idx:03d}"

        # Image size
        img = Image.open(img_path)
        w, h = img.size

        # --- Write image ---
        img_data = img_path.read_bytes()
        info = tarfile.TarInfo(name=f"{key}.png")
        info.size = len(img_data)
        tar.addfile(info, io.BytesIO(img_data))

        # --- Write mask ---
        if mask_path.exists():
            mask_data = mask_path.read_bytes()
            info = tarfile.TarInfo(name=f"{key}.mask.png")
            info.size = len(mask_data)
            tar.addfile(info, io.BytesIO(mask_data))

        # --- Write metadata JSON ---
        meta = {
            "bbox": bbox,           # [x1, y1, x2, y2] in pixels
            "label": label,         # "benign" or "malignant"
            "case_id": case_id,
            "slice_idx": slice_idx,
            "size": [h, w],         # [height, width]
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        info = tarfile.TarInfo(name=f"{key}.json")
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))

        sample_count += 1

    if tar:
        tar.close()

    print(f"  → {sample_count} samples in {shard_idx} shard(s)")


def main():
    parser = argparse.ArgumentParser(description="Pack ABUS 2D data into WebDataset shards")
    parser.add_argument("--data-dir", type=Path, default=Path("/Volumes/Autzoko/ABUS_2D"))
    parser.add_argument("--out-dir", type=Path, default=Path("shards_abus"))
    parser.add_argument("--samples-per-shard", type=int, default=500)
    args = parser.parse_args()

    splits = [
        ("train", "Train_metadata.csv", "Train"),
        ("val", "Validation_metadata.csv", "Validation"),
        ("test", "Test_metadata.csv", "Test"),
    ]

    for split, csv_name, subdir in splits:
        csv_path = args.data_dir / csv_name
        data_dir = args.data_dir / subdir
        if csv_path.exists() and data_dir.exists():
            pack_split(data_dir, csv_path, args.out_dir, split, args.samples_per_shard)
        else:
            print(f"[{split}] SKIP — missing {csv_path} or {data_dir}")

    print("\nDone. Transfer shards_abus/ to HPC.")


if __name__ == "__main__":
    main()
