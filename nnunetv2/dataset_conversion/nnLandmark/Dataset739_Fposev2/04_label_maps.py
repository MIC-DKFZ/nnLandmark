#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from typing import Dict, List

def draw_cube(seg, center, half=1, label=1):
    x, y, z = (int(round(c)) for c in center)
    xs = slice(max(x-half, 0), x+half+1)
    ys = slice(max(y-half, 0), y+half+1)
    zs = slice(max(z-half, 0), z+half+1)
    seg[xs, ys, zs] = label

def find_image(base: Path, case_key: str):
    # Prefer case_key_0000.nii.gz then any case_key_*.nii* then case_key.nii.gz
    for ext in (".nii.gz", ".nii"):
        p = base / f"{case_key}_0000{ext}"
        if p.exists():
            return p
    candidates = sorted(base.glob(f"{case_key}_*.nii.gz")) + sorted(base.glob(f"{case_key}_*.nii"))
    if candidates:
        return candidates[0]
    for ext in (".nii.gz", ".nii"):
        p = base / f"{case_key}{ext}"
        if p.exists():
            return p
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",   default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/imagesTs",
                   help="Pose volumes folder")
    p.add_argument("--landmarks", default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/all_landmarks_voxel.json",
                   help="JSON of landmarks")
    p.add_argument("--name2label", default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/name_to_label.json",
                   help="JSON: landmark_<n> → int label  (default: derive on the fly)")
    p.add_argument("--output",    default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/labelsTs/",
                   help="where to write segmaps")
    p.add_argument("--half", type=int, default=1,
                   help="Half cube size (default: 1 → 3×3×3)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing label files if they exist.")
    args = p.parse_args()

    base = Path(args.base)
    out  = Path(args.output); out.mkdir(exist_ok=True, parents=True)

    # 1) load landmarks JSON
    all_lms = json.loads(Path(args.landmarks).read_text())

    # 2) load or derive name→label
    if args.name2label:
        name_to_label = json.loads(Path(args.name2label).read_text())
    else:
        # simple default fallback
        name_to_label = {
            f"landmark_{i}": i
            for i in range(1, 22)
        }

    print(f"Using {len(name_to_label)} labels")

    for case_key, lms in all_lms.items():
        # find image
        img_path = find_image(base, case_key)
        if img_path is None:
            print(f"[WARN] no image for {case_key}")
            continue

        out_path = out / f"{case_key}.nii.gz"
        if out_path.exists() and not args.overwrite:
            print(f"SKIP_EXISTS {out_path.name}: file already exists (use --overwrite to replace)")
            continue

        img = nib.load(str(img_path))
        # allocate segmentation using shape info from header (avoid loading image data)
        shape = img.header.get_data_shape()[:3]
        if len(shape) != 3:
            print(f"[WARN] unexpected image shape for {case_key}: {shape}")
            continue
        seg = np.zeros(shape, dtype=np.uint8)

        found_any = False
        for name, label in name_to_label.items():
            if name in lms:
                coords = lms[name]
                draw_cube(seg, coords, half=args.half, label=label)
                found_any = True

        if not found_any:
            print(f"[WARN] no landmarks for {case_key}, writing empty label (use --overwrite to replace existing)")
            # still write empty seg unless you prefer to skip — currently we write it
        # prepare header: copy image header but ensure dtype is uint8
        hdr = img.header.copy()
        hdr.set_data_dtype(np.uint8)
        seg_img = nib.Nifti1Image(seg, img.affine, hdr)
        nib.save(seg_img, str(out_path))
        print(f"Wrote {out_path.name}")

if __name__=="__main__":
    main()