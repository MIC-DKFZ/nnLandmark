##!/usr/bin/env python3
"""mml_segmentation_from_landmarks.py

Paint cubic markers into the MMLD volumes using the consolidated landmark JSONs.

Inputs
------
* ``mmld_dataset_all_landmarks.json``   – produced by *mml_landmarks_and_labels.py*
* ``mmld_dataset_name_to_label.json``   – stable label map (same order)

The script locates the corresponding ``*_volume.nrrd`` in *train*, *val* or
*test* folders, writes a label map with identical header next to the nnU‑Net
raw folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import nrrd

# ------------------------------------------------------------------ helpers

def draw_cube(seg: np.ndarray, center: list[float], *, half: int, label: int):
    x, y, z = (int(round(c)) for c in center)
    seg[max(x-half, 0):x+half+1,
        max(y-half, 0):y+half+1,
        max(z-half, 0):z+half+1] = label

# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/path/to/Project_nnLandmark/data/mmld_dataset/",
                    help="mmld_dataset root (contains train/val/test)")
    ap.add_argument("--landmarks", default="/path/to/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset733_MML/all_landmarks_voxel.json",
                    help="case → { landmark_<n>: [i,j,k] }")
    ap.add_argument("--name2label", default="/path/to/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset733_MML/name_to_label.json",
                    help="landmark_<n> → integer label (order preserved)")
    ap.add_argument("--output", default="/path/to/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset733_MML/labelsTr/",
                    help="Destination folder for the segmentation maps")
    ap.add_argument("--half", type=int, default=1,
                    help="Half cube size (default 1 → 3×3×3)")
    args = ap.parse_args()

    base = Path(args.base)
    out  = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    all_lms     = json.loads(Path(args.landmarks).read_text())
    name2label  = json.loads(Path(args.name2label).read_text())
    print(f"Using {len(name2label)} distinct labels")

    splits = ["train"]

    for case_id, lms in all_lms.items():                # e.g. 'MML_sub-12345'
        # ------------------------------------------------ locate volume
        vol_path = None
        for split in splits:
            cand = base / split / f"{case_id}_volume.nrrd"
            if cand.exists():
                vol_path = cand
                break
        if vol_path is None:
            print(f"[WARN] volume not found for {case_id}")
            continue

        img, hdr = nrrd.read(str(vol_path))
        seg = np.zeros(img.shape, dtype=np.uint8)

        # ------------------------------------------------ paint cubes
        for name, label in name2label.items():
            if name in lms:
                draw_cube(seg, lms[name], half=args.half, label=label)
            else:
                print(f"[WARN] landmark '{name}' missing in {case_id}")

        # ------------------------------------------------ write label map
        out_file = out / f"{case_id}.nrrd"
        nrrd.write(str(out_file), seg, header=hdr)
        print(f"✅  {out_file.name}")


if __name__ == "__main__":
    main()
