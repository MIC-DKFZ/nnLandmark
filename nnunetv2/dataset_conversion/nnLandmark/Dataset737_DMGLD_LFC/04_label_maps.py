#!/usr/bin/env python3
from __future__ import annotations  # optional
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib


def draw_cube(arr: np.ndarray, center_ijk, half: int, label: int):
    i, j, k = (int(round(c)) for c in center_ijk)  # voxel indices (i,j,k)
    i0, i1 = max(i - half, 0), min(i + half + 1, arr.shape[0])
    j0, j1 = max(j - half, 0), min(j + half + 1, arr.shape[1])
    k0, k1 = max(k - half, 0), min(k + half + 1, arr.shape[2])
    if i0 < i1 and j0 < j1 and k0 < k1:
        arr[i0:i1, j0:j1, k0:k1] = label


def infer_images_dir(images_root: Path) -> Path:
    # Accept either imagesTr/imagesTs under root, or a single directory of images
    if (images_root / "imagesTr").is_dir():
        return images_root / "imagesTr"
    if (images_root / "imagesTs").is_dir():
        return images_root / "imagesTs"
    return images_root


def find_image_for_case(images_dir: Path, case: str) -> Optional[Path]:
    # Prefer {case}_0000.nii.gz → then any {case}_*.nii.gz → then {case}_*.nii
    candidates = [
        images_dir / f"{case}_0000.nii.gz",
        images_dir / f"{case}_0000.nii",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: first file starting with case_
    for p in sorted(images_dir.glob(f"{case}_*.nii.gz")) + sorted(images_dir.glob(f"{case}_*.nii")):
        return p
    # Last resort: exact {case}.nii.gz or {case}.nii
    for p in [images_dir / f"{case}.nii.gz", images_dir / f"{case}.nii"]:
        if p.exists():
            return p
    return None


def derive_name_to_label(names: list[str]) -> dict[str, int]:
    import re
    out = {}
    for n in sorted(names, key=lambda s: int(re.search(r"(\d+)$", s).group(1)) if re.search(r"(\d+)$", s) else s):
        m = re.search(r"(\d+)$", n)
        out[n] = int(m.group(1)) if m else len(out) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate label masks from nnLandmark voxel JSONs and imagesTr/imagesTs.")
    ap.add_argument("--images", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/imagesTs", help="Path to images root (either the dataset root containing imagesTr/imagesTs, or a single images dir).")
    ap.add_argument("--landmarks", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/all_landmarks_voxel_test.json", help="Path to landmarks JSON (e.g., all_landmarks_voxel_train.json or test).")
    ap.add_argument("--name2label", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/name_to_label.json", help="Optional path to name_to_label.json; if omitted, derive from landmark names.")
    ap.add_argument("--output", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/labelsTs", help="Output directory for label masks (e.g., labelsTr or labelsTs).")
    ap.add_argument("--half", type=int, default=1, help="Half cube size (1 => 3x3x3).")
    ap.add_argument("--dtype", default="uint8", choices=["uint8", "uint16", "int16", "int32"], help="Output label volume dtype.")
    args = ap.parse_args()

    images_root = Path(args.images).resolve()
    images_dir = infer_images_dir(images_root)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load landmarks: { case: { landmark_X: [i,j,k], ... }, ... }
    landmarks = json.loads(Path(args.landmarks).read_text())

    # Load or derive name_to_label
    if args.name2label:
        name_to_label = json.loads(Path(args.name2label).read_text())
    else:
        # union of all names present across cases
        all_names = set()
        for _, d in landmarks.items():
            all_names.update(d.keys())
        name_to_label = derive_name_to_label(sorted(all_names))

    print(f"Images dir: {images_dir}")
    print(f"Landmarks: {len(landmarks)} cases")
    print(f"Name->label entries: {len(name_to_label)}")

    np_dtype = getattr(np, args.dtype)

    for case, lm_dict in landmarks.items():
        img_path = find_image_for_case(images_dir, case)
        if img_path is None:
            print(f"[WARN] Image not found for case '{case}' in '{images_dir}'")
            continue

        img = nib.load(str(img_path))
        seg = np.zeros(img.shape, dtype=np_dtype)

        # Stamp cubes for each present landmark name
        for name, label in name_to_label.items():
            coord = lm_dict.get(name, None)
            if coord is None:
                continue
            draw_cube(seg, coord, half=args.half, label=int(label))

        out_path = out_dir / f"{case}.nii.gz"
        nib.save(nib.Nifti1Image(seg, img.affine, img.header), str(out_path))
        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
