##!/usr/bin/env python3
"""mml_segmentation_from_landmarks.py

Paint cubic markers into the MMLD volumes using the consolidated landmark JSONs.

Inputs
------
* ``mmld_dataset_all_landmarks.json``   – produced by *mml_landmarks_and_labels.py*
* optional name→label JSON; by default the labels are derived from the landmark JSON
  exactly like 06_generate_dataset_json.py does, so they match dataset.json

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


def derive_name_to_label(all_lms: dict) -> dict[str, int]:
    """Labels 1..N in order of first appearance, the same rule as 06_generate_dataset_json.py."""
    name2label = {}
    for case_lms in all_lms.values():
        for name in case_lms:
            if name not in name2label:
                name2label[name] = len(name2label) + 1
    return name2label

# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/path/to/Project_nnLandmark/data/mmld_dataset/",
                    help="mmld_dataset root (contains train/val/test)")
    ap.add_argument("--landmarks", default="/path/to/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset733_MML/all_landmarks_voxel.json",
                    help="case → { landmark_<n>: [i,j,k] }")
    ap.add_argument("--name2label", default=None,
                    help="optional JSON landmark_<n> → integer label; if omitted, derived from --landmarks "
                         "like 06_generate_dataset_json.py does")
    ap.add_argument("--output", default="/path/to/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset733_MML/labelsTr/",
                    help="Destination folder for the segmentation maps")
    ap.add_argument("--half", type=int, default=1,
                    help="Half cube size (default 1 → 3×3×3)")
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="mmld_dataset splits to take volumes from (default: train); e.g. --splits test for labelsTs")
    args = ap.parse_args()

    base = Path(args.base)
    out  = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    all_lms     = json.loads(Path(args.landmarks).read_text())
    if args.name2label:
        name2label = json.loads(Path(args.name2label).read_text())
    else:
        name2label = derive_name_to_label(all_lms)
    print(f"Using {len(name2label)} distinct labels")

    splits = args.splits

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
