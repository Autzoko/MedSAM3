#!/usr/bin/env python3
"""Pack 3D BIRADS ABUS data into 2D coronal-slice WebDataset shards.

Reads NIfTI volumes + JSON label annotations, slices along the coronal axis
(axis 1), and packs into .tar shards compatible with evaluate_webdataset.py
and infer_webdataset.py.

Shard format:
    birads-000000.tar
      ├── 1_361234/0283.png     (coronal slice at y=283)
      ├── 1_361234/0283.txt     (GT label: cls_id x1 y1 x2 y2)
      ├── 1_361234/0284.png     (slice without annotation → no .txt)
      └── ...

Usage:
    python pack_birads_shards.py \
        --data-dir "/Volumes/Autzoko/Dataset/Ultrasound/已标注及BI-rads分类20260123" \
        --out-dir shards_birads
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def load_labels(tar_path: Path) -> dict:
    """Load label JSON from a *_Label.tar file."""
    with tarfile.open(str(tar_path), "r") as tar:
        for member in tar:
            if member.name.endswith(".json"):
                return json.loads(tar.extractfile(member).read())
    return {}


def extract_coronal_bboxes(label_data: dict) -> dict[int, list[list[float]]]:
    """Extract coronal (SliceType=1) bounding boxes.

    Returns: {voxel_y_index: [[x1, y1, x2, y2], ...]}  (y = z in 2D slice)
    Coordinates are in voxel space (pixels).
    """
    bboxes_by_slice: dict[int, list[list[float]]] = {}
    spacing_y = label_data.get("FileInfo", {}).get("Spacing", [1, 1, 1])[1]

    bbox_models = (
        label_data.get("Models", {}).get("BoundingBoxLabelModel", []) or []
    )
    for bb in bbox_models:
        if bb["SliceType"] != 1:  # only coronal
            continue

        # Voxel slice index from physical Y coordinate
        slice_idx = int(round(bb["p1"][1] / spacing_y))

        # Bbox in coronal plane: X and Z voxel coordinates
        x1 = min(bb["p1"][0], bb["p2"][0])
        x2 = max(bb["p1"][0], bb["p2"][0])
        z1 = min(bb["p1"][2], bb["p2"][2])
        z2 = max(bb["p1"][2], bb["p2"][2])

        if slice_idx not in bboxes_by_slice:
            bboxes_by_slice[slice_idx] = []
        bboxes_by_slice[slice_idx].append([x1, z1, x2, z2])

    return bboxes_by_slice


def slice_to_png_bytes(vol: np.ndarray, y_idx: int) -> bytes:
    """Extract coronal slice at y_idx and return as PNG bytes."""
    # vol shape: (W, H, D) → coronal slice = vol[:, y_idx, :] → (W, D)
    slc = vol[:, y_idx, :].astype(np.float64)

    # Normalize to 0-255
    vmin, vmax = slc.min(), slc.max()
    if vmax > vmin:
        slc = (slc - vmin) / (vmax - vmin) * 255.0
    else:
        slc = np.zeros_like(slc)

    # Transpose to (D, W) so the image is (height=D, width=W)
    img = Image.fromarray(slc.T.astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def format_label(bboxes: list[list[float]], label_id: int = 1) -> str:
    """Format bboxes as label text: cls_id x1 y1 x2 y2 per line."""
    lines = []
    for box in bboxes:
        lines.append(f"{label_id} {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}")
    return "\n".join(lines)


def pack_shards(
    data_dir: Path,
    out_dir: Path,
    samples_per_shard: int = 2000,
    slice_every: int = 1,
    only_annotated: bool = False,
) -> None:
    """Pack all BIRADS volumes into WebDataset shards."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all NIfTI files
    nii_files = sorted(data_dir.rglob("*.nii"))
    print(f"Found {len(nii_files)} NIfTI volumes")

    shard_idx = 0
    sample_count = 0
    total_samples = 0
    tar = None

    for nii_path in nii_files:
        case_id = nii_path.stem  # e.g., "1_361234"

        # Find label file
        label_tar = nii_path.parent / f"{case_id}_nii_Label.tar"
        label_json = nii_path.parent / f"{case_id}_nii_Label.json"

        label_data = {}
        if label_tar.exists():
            label_data = load_labels(label_tar)
        elif label_json.exists():
            with open(label_json) as f:
                label_data = json.load(f)

        coronal_bboxes = extract_coronal_bboxes(label_data) if label_data else {}

        # Load volume
        try:
            vol = nib.load(str(nii_path)).get_fdata()
        except Exception as e:
            print(f"  SKIP {nii_path.name}: {e}")
            continue

        n_slices = vol.shape[1]  # coronal axis
        n_written = 0

        for y_idx in range(0, n_slices, slice_every):
            has_label = y_idx in coronal_bboxes
            if only_annotated and not has_label:
                continue

            # Open new shard if needed
            if sample_count % samples_per_shard == 0:
                if tar:
                    tar.close()
                tar_path = out_dir / f"birads-{shard_idx:06d}.tar"
                tar = tarfile.open(str(tar_path), "w")
                shard_idx += 1

            key = f"{case_id}/{y_idx:04d}"

            # Write PNG
            png_bytes = slice_to_png_bytes(vol, y_idx)
            info = tarfile.TarInfo(name=f"{key}.png")
            info.size = len(png_bytes)
            tar.addfile(info, io.BytesIO(png_bytes))

            # Write label (if annotated)
            if has_label:
                label_text = format_label(coronal_bboxes[y_idx])
                label_bytes = label_text.encode("utf-8")
                info = tarfile.TarInfo(name=f"{key}.txt")
                info.size = len(label_bytes)
                tar.addfile(info, io.BytesIO(label_bytes))

            sample_count += 1
            n_written += 1

        total_samples += n_written
        n_annotated = len(coronal_bboxes)
        print(f"  {case_id}: {n_slices} slices, {n_annotated} annotated, {n_written} written")

    if tar:
        tar.close()

    print(f"\nDone: {total_samples} slices in {shard_idx} shard(s) → {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Pack 3D BIRADS ABUS data into 2D coronal WebDataset shards"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/Volumes/Autzoko/Dataset/Ultrasound/已标注及BI-rads分类20260123"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("shards_birads"))
    parser.add_argument("--samples-per-shard", type=int, default=2000)
    parser.add_argument(
        "--slice-every", type=int, default=1,
        help="Take every N-th coronal slice (default: 1 = all slices)",
    )
    parser.add_argument(
        "--only-annotated", action="store_true",
        help="Only pack slices that have GT bounding box annotations",
    )
    args = parser.parse_args()

    pack_shards(
        args.data_dir,
        args.out_dir,
        args.samples_per_shard,
        args.slice_every,
        args.only_annotated,
    )


if __name__ == "__main__":
    main()
