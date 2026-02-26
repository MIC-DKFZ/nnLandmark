#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Copy & rename all pose‐dataset volumes into one folder")
    parser.add_argument("--base", default="/home/a332l/dev/Project_nnLandmark/data/landmark_datasets/face_dataset/volume/",
                        help="Pose dataset volumes directory")
    parser.add_argument("--target", default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset736_Fetal_face/imagesAll/", help="Target directory to copy renamed images into")
    args = parser.parse_args()

    vol_dir = Path(os.path.expanduser(args.base))
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    # find both .nii and .nii.gz
    img_files = sorted(vol_dir.glob("*.nii.gz"))
    for img in img_files:
        ext = "".join(img.suffixes)  # ".nii" or ".nii.gz"
        stem = img.name[:-len(ext)]
        new_name = f"{stem}_0000{ext}"
        shutil.copy(img, target / new_name)
        print(f"Copied {img.name} → {new_name}")

if __name__ == "__main__":
    main()
