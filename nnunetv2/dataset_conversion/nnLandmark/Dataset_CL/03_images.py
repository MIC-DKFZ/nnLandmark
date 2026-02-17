#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from PIL import Image

def main():
    ap = argparse.ArgumentParser(
        description="Copy 2D images, convert to grayscale (1-channel), rename to nnU-Net style: <prefix><stem>_0000.bmp"
    )
    ap.add_argument(
        "--images_dir",
        default="/home/a332l/dev/Project_nnLandmark/data/CL-Detection2024_Accessible_Data/Training_Set/images/",
        type=Path,
    )
    ap.add_argument(
        "--target",
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/imagesTr/",
        type=Path,
    )
    ap.add_argument("--prefix", default="", help="Prefix for case IDs.")
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=["*.bmp", "*.BMP", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"],
        help="Glob patterns to include.",
    )
    args = ap.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)

    files = []
    for pat in args.patterns:
        files.extend(sorted(args.images_dir.glob(pat)))

    if not files:
        print(f"[INFO] No files matched in {args.images_dir} with patterns {args.patterns}")
        return

    done = 0
    for src in files:
        stem = src.stem
        out_name = f"{args.prefix}{stem}_0000.bmp"
        dst = args.target / out_name

        # Convert to grayscale and save
        img = Image.open(src).convert("L")  # "L" = grayscale
        img.save(dst)
        done += 1
        print(f"Saved {src.name} → {out_name}, shape: (1, H, W)")

    print(f"\n✅ {done} files saved to {args.target}")

if __name__ == "__main__":
    main()
