#!/usr/bin/env python3
"""
Create label maps (nrrd) for PDDCA dataset.

- Reads canonical voxel landmarks JSON (all_landmarks_voxel.json).
- Finds corresponding image in imagesTr named {case}_0000.nrrd/.nhdr and creates label map
  with the same filename but without the "_0000" token (e.g. case.nrrd).
- Preserves useful header keys (space directions/origin/kinds) when writing nrrd.
- Draws small cubic markers (half radius configurable) at voxel coords using name_to_label mapping.
"""
import argparse
import json
import re
from pathlib import Path
from collections import OrderedDict

import numpy as np

try:
    import nrrd  # type: ignore
except Exception:
    nrrd = None

# Helpers
def draw_cube(seg: np.ndarray, center, *, half: int, label: int):
    x, y, z = (int(round(c)) for c in center)
    seg[
        max(x - half, 0) : x + half + 1,
        max(y - half, 0) : y + half + 1,
        max(z - half, 0) : z + half + 1,
    ] = label

def _choose_dtype(max_label: int):
    return np.uint16 if max_label > 255 else np.uint8

def _strip_0000(name: str) -> str:
    # remove trailing _0000 before extension
    return re.sub(r"_0000(?=\.)", "", name)

def _collect_header_for_write(header: dict):
    if not header:
        return {}
    out = {}
    for key in ("space directions", "space_directions", "space-directions",
                "space origin", "space_origin", "space-origin", "space origin (mm)",
                "kinds", "space"):
        if key in header:
            out[key] = header[key]
    # set content/encoding hints if present
    for key in ("encoding", "type"):
        if key in header:
            out.setdefault(key, header[key])
    return out

def main():
    p = argparse.ArgumentParser(description="Create nrrd label maps from voxel landmark JSON")
    p.add_argument("--images", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA/imagesTr",
                   help="imagesTr dir containing {case}_0000.nrrd")
    p.add_argument("--landmarks", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA/all_landmarks_voxel.json",
                   help="JSON: case -> { landmark_X: [i,j,k], ... } (voxel coords)")
    p.add_argument("--name2label", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA/name_to_label.json",
                   help="JSON: canonical name -> int label")
    p.add_argument("--output", default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA/labelsTr",
                   help="output directory for label nrrd files")
    p.add_argument("--half", type=int, default=1, help="half-cube size (half=1 => 3x3x3 cube)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing label files")
    args = p.parse_args()

    if nrrd is None:
        raise SystemExit("pynrrd required. Install: pip install pynrrd")

    images_dir = Path(args.images)
    landmarks_path = Path(args.landmarks)
    name2label_path = Path(args.name2label)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not landmarks_path.exists():
        raise SystemExit(f"landmarks JSON not found: {landmarks_path}")

    all_lms = json.loads(landmarks_path.read_text())

    if name2label_path.exists():
        name_to_label = json.loads(name2label_path.read_text())
    else:
        # derive mapping from present keys
        canonical_names = set()
        for case_lms in all_lms.values():
            canonical_names.update(case_lms.keys())
        name_to_label = {name: i for i, name in enumerate(sorted(canonical_names), start=1)}

    max_label = max(name_to_label.values()) if name_to_label else 1
    dtype = _choose_dtype(max_label)

    print(f"Found {len(all_lms)} cases, {len(name_to_label)} labels, dtype={dtype.__name__}")

    for case_key, lms in all_lms.items():
        # find image file in images_dir named {case_key}_0000.*
        img_candidates = []
        if images_dir.exists():
            img_candidates = sorted(images_dir.glob(f"{case_key}_0000*"))
        if not img_candidates:
            print(f"[WARN] image for case {case_key} not found in {images_dir}, skipping")
            continue
        img_path = img_candidates[0]
        try:
            img_data, img_header = nrrd.read(str(img_path))
        except Exception as e:
            print(f"[ERROR] failed to read image {img_path}: {e}; skipping")
            continue

        shape = img_data.shape
        seg = np.zeros(shape, dtype=dtype)

        # draw each landmark present in this case
        for name, label in name_to_label.items():
            if name in lms:
                center = lms[name]
                try:
                    draw_cube(seg, center, half=args.half, label=int(label))
                except Exception as e:
                    print(f"[WARN] failed to draw {name} in {case_key}: {e}")

        out_name = _strip_0000(img_path.name)
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists (use --overwrite to replace)")
            continue

        write_header = _collect_header_for_write(img_header)
        try:
            nrrd.write(str(out_path), seg, header=write_header)
            print(f"WROTE {out_path.name}")
        except Exception as e:
            print(f"[ERROR] writing {out_path}: {e}")

if __name__ == "__main__":
    main()