#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Copy & rename all NRRD volumes from mmld_dataset into one folder")
    parser.add_argument("--base", default="~/dev/Project_SoftDiceLoss/data/mmld_dataset",
                        help="Base mmld_dataset directory containing train/val/test")
    parser.add_argument("--target", default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset734_MML/imagesTs/", help="Target directory to copy renamed images into")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="Which splits to scan (default: train val test)")
    args = parser.parse_args()

    base = Path(os.path.expanduser(args.base))
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for img in sorted(split_dir.glob("*_volume.nrrd")):
            # capture full extension
            ext = "".join(img.suffixes)     # ".nrrd"
            stem = img.name[:-len(ext)]     # e.g. "case123_volume"
            # remove the "_volume" suffix
            new_stem = stem.replace("_volume", "")
            new_name = f"{new_stem}_0000{ext}"
            shutil.copy(img, target / new_name)
            print(f"Copied {img.relative_to(base)} → {new_name}")

if __name__ == "__main__":
    main()
