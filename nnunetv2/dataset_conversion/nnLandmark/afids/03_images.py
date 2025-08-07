#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Copy & rename all AFIDs NIfTI images into one folder"
    )
    parser.add_argument(
        "--base",
        default="~/dev/Project_SoftDiceLoss/data/afids-data/data/datasets",
        help="AFIDs datasets root (contains multiple sub-datasets)"
    )
    parser.add_argument(
        "--target",
        default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset733_Afids/imagesAll/",
        help="Target directory to copy renamed images into"
    )
    args = parser.parse_args()

    base = Path(os.path.expanduser(args.base))
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    for ds in sorted(base.iterdir()):
        if not ds.is_dir(): continue

        # find all .nii and .nii.gz under sub-*/anat
        img_files = sorted(ds.glob("sub-*/anat/*.nii")) + \
                    sorted(ds.glob("sub-*/anat/*.nii.gz"))
        for img in img_files:
            dataset = ds.name                       # e.g. "dataset01"
            sub_id  = img.parents[1].name           # e.g. "sub-103111"
            ext     = "".join(img.suffixes)         # ".nii" or ".nii.gz"
            new_name = f"{dataset}_{sub_id}_0000{ext}"
            shutil.copy(img, target / new_name)
            print(f"Copied {img.relative_to(base)} → {new_name}")

if __name__ == "__main__":
    main()
