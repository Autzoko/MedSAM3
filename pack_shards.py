#!/usr/bin/env python3
"""Pack BIrads dataset into WebDataset .tar shards.

Each sample is one 2D slice with its label, keyed by {case_id}/{slice_idx}.
Shards are grouped by case so all slices of a 3D volume stay together.

Usage:
    python pack_shards.py
    python pack_shards.py --data-root /path/to/BIrads --output shards/ --max-per-shard 500

Output structure:
    shards/
      birads-000000.tar
      birads-000001.tar
      ...
      manifest.json   (case→shard mapping + metadata)
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

DATA_ROOT = Path("/Volumes/Lang/Research/Data/3D Ultrasound/processed/BIrads")


def make_sample(case_id: str, slice_idx: str, img_path: Path, lbl_path: Path) -> dict:
    """Build a WebDataset sample dict."""
    key = f"{case_id}/{slice_idx}"

    img_bytes = img_path.read_bytes()
    lbl_text = lbl_path.read_text().strip() if lbl_path.exists() else "0 0 0 0 0"

    # Metadata JSON
    meta = {"case_id": case_id, "slice_idx": slice_idx}

    return {
        "__key__": key,
        "png": img_bytes,
        "txt": lbl_text.encode("utf-8"),
        "json": json.dumps(meta).encode("utf-8"),
    }


def write_tar(samples: list[dict], tar_path: Path):
    """Write samples to a .tar file in WebDataset format."""
    with tarfile.open(str(tar_path), "w") as tar:
        for sample in samples:
            key = sample["__key__"]
            for ext, data in sample.items():
                if ext == "__key__":
                    continue
                if isinstance(data, str):
                    data = data.encode("utf-8")
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description="Pack BIrads into WebDataset shards")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("shards"))
    parser.add_argument("--max-per-shard", type=int, default=500,
                        help="Max slices per shard (cases are not split across shards)")
    args = parser.parse_args()

    images_dir = args.data_root / "images"
    labels_dir = args.data_root / "labels"
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy summary.csv
    summary_src = args.data_root / "summary.csv"
    if summary_src.exists():
        (out_dir / "summary.csv").write_bytes(summary_src.read_bytes())

    case_ids = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    print(f"Found {len(case_ids)} cases")

    manifest = {"shards": [], "cases": {}}
    shard_idx = 0
    current_samples: list[dict] = []
    current_shard_cases: list[str] = []

    def flush_shard():
        nonlocal shard_idx, current_samples, current_shard_cases
        if not current_samples:
            return
        tar_name = f"birads-{shard_idx:06d}.tar"
        tar_path = out_dir / tar_name
        write_tar(current_samples, tar_path)
        manifest["shards"].append({
            "name": tar_name,
            "n_samples": len(current_samples),
            "cases": current_shard_cases,
        })
        for cid in current_shard_cases:
            manifest["cases"][cid]["shard"] = tar_name
        print(f"  Written {tar_name}: {len(current_samples)} slices, "
              f"{len(current_shard_cases)} cases")
        shard_idx += 1
        current_samples = []
        current_shard_cases = []

    for case_id in tqdm(case_ids, desc="Packing cases"):
        img_dir = images_dir / case_id
        lbl_dir = labels_dir / case_id

        slice_files = sorted(img_dir.glob("slice_*.png"))
        if not slice_files:
            continue

        case_samples = []
        n_positive = 0
        for img_path in slice_files:
            idx = img_path.stem.replace("slice_", "")
            lbl_path = lbl_dir / f"slice_{idx}.txt"
            sample = make_sample(case_id, idx, img_path, lbl_path)
            case_samples.append(sample)

            # Check if positive
            lbl_text = lbl_path.read_text().strip() if lbl_path.exists() else ""
            for line in lbl_text.splitlines():
                parts = line.split()
                if len(parts) >= 5 and int(float(parts[0])) != 0:
                    n_positive += 1
                    break

        manifest["cases"][case_id] = {
            "n_slices": len(case_samples),
            "n_positive": n_positive,
        }

        # Would adding this case exceed shard limit?
        if current_samples and len(current_samples) + len(case_samples) > args.max_per_shard:
            flush_shard()

        current_samples.extend(case_samples)
        current_shard_cases.append(case_id)

    # Final shard
    flush_shard()

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_slices = sum(m["n_slices"] for m in manifest["cases"].values())
    print(f"\nDone: {len(manifest['shards'])} shards, "
          f"{len(case_ids)} cases, {total_slices} total slices")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
