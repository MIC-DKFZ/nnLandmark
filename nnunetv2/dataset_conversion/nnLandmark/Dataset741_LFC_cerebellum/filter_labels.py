#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

# mapping: keep only these original label values and remap to 1..6
KEEP_MAP = {5: 1, 6: 2, 9: 3, 10: 4, 11: 5, 12: 6}
KEEP_VALUES = set(KEEP_MAP.keys())


def process_file(p: Path, out_path: Path) -> None:
    img = nib.load(str(p))
    # read data as integer array
    arr = np.array(img.dataobj, copy=True)
    arr = arr.astype(np.int32, copy=False)

    out = np.zeros_like(arr, dtype=np.int16)

    total_kept = 0
    for orig, new in KEEP_MAP.items():
        mask = arr == orig
        c = int(mask.sum())
        if c:
            out[mask] = int(new)
            total_kept += c

    # prepare header: force integer dtype
    hdr = img.header.copy()
    hdr.set_data_dtype(np.int16)

    out_img = nib.Nifti1Image(out, img.affine, hdr)
    out_img.to_filename(str(out_path))

    print(f"{p.name}: kept_voxels={total_kept}, written={out_path.name}")


def find_niftis(d: Path):
    for ext in ("*.nii", "*.nii.gz"):
        for p in sorted(d.glob(ext)):
            yield p


def main():
    p = argparse.ArgumentParser(description="Filter landmark label nifti(s) in-place: keep labels 5,6,9,10,11,12 and remap to 1..6.")
    p.add_argument("--dataset_root", default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset741_LFC_cerebellum", type=Path, help="root containing labelsTr/labelsTs or a labels dir")
    args = p.parse_args()

    root = args.dataset_root
    if root is None or not root.exists():
        print("dataset_root not found or not specified:", root, file=sys.stderr)
        sys.exit(1)

    label_dirs = []
    if root.name in ("labelsTr", "labelsTs"):
        label_dirs = [root]
    else:
        for sub in ("labelsTr", "labelsTs"):
            d = root / sub
            if d.exists():
                label_dirs.append(d)

    if not label_dirs:
        print("No labelsTr/labelsTs found under", root, file=sys.stderr)
        sys.exit(1)

    for ld in label_dirs:
        for nif in find_niftis(ld):
            # overwrite input file in-place (no backup, no suffix)
            process_file(nif, nif)


if __name__ == "__main__":
    main()