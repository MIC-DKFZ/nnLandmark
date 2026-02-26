#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

def copy_split(src_img_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(src_img_dir.glob("*.nii")) + sorted(src_img_dir.glob("*.nii.gz"))
    for p in imgs:
        case = p.stem.replace(".nii", "")
        ext = "".join(p.suffixes)
        new_name = f"{case}_0000{ext}"
        shutil.copy2(p, dst_dir / new_name)
        print(f"Copied {p.name} -> {new_name}")

def main():
    ap = argparse.ArgumentParser(description="Copy/rename images into nnLandmark layout")
    ap.add_argument("--src", required=True, help="Source root with train/ and valid/")
    ap.add_argument("--dst", required=True, help="Target dataset root with imagesTr/imagesTs")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    copy_split(src / "train", dst / "imagesTr")
    copy_split(src / "valid", dst / "imagesTs")
    print("Done.")

if __name__ == "__main__":
    main()
