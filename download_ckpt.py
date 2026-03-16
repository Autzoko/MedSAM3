#!/usr/bin/env python3
"""Download the Medical-SAM3 checkpoint from HuggingFace.

Usage:
    python download_ckpt.py                      # default → checkpoints/checkpoint.pt
    python download_ckpt.py --out /other/path
    python download_ckpt.py --force               # re-download even if it exists

Requires: pip install huggingface_hub tqdm
If the repo is gated, run `huggingface-cli login` first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "ChongCong/Medical-SAM3"
FILENAME = "checkpoint.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Medical-SAM3 checkpoint")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Directory to save checkpoint.pt into (default: ./checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()

    dest_dir: Path = args.out
    dest_file = dest_dir / FILENAME

    if dest_file.exists() and not args.force:
        size_gb = dest_file.stat().st_size / (1024**3)
        print(f"Checkpoint already exists: {dest_file} ({size_gb:.1f} GB) — skipping.")
        print("Use --force to re-download.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {REPO_ID}/{FILENAME} → {dest_dir}/")
    print("(This is ~10 GB — grab a coffee.)\n")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"\n✓ Saved to {path} ({size_gb:.1f} GB)")
    except Exception as exc:
        print(f"\nERROR: Download failed — {exc}")
        print("If the repo is gated, run:  huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    main()
