#!/usr/bin/env python3
"""mml_landmarks_and_labels.py

Collect voxel‐space landmarks from the *mmld_dataset* folder and write three
JSON files side‑by‑side:

* ``mmld_dataset_all_landmarks.json``   – case → { landmark_<n>: [i,j,k] }
* ``mmld_dataset_name_to_label.json``   – landmark_<n> → label (order = first appearance)
* ``mmld_dataset_spacing.json``         – case → { image_spacing, annotation_spacing }

Background (label 0) is reserved; therefore if a landmark key ends in a
numeric suffix “n” the assigned label will be ``n+1`` when ``n`` starts at 0.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import nrrd

# --------------------------------------------------------------------- helpers

def spacing_from_header(header):
    """Extract voxel spacing from NRRD header (returns tuple or raises)."""
    if header.get("space directions"):
        return tuple(float(np.linalg.norm(v)) if v is not None else 1.0 for v in header["space directions"])
    if "spacing" in header:
        return tuple(float(x) for x in header["spacing"])
    raise ValueError("No spacing info in NRRD header")

# --------------------------------------------------------------------- main

def main():
    base = Path.home() / "dev" / "Project_nnLandmark" / "data" / "mmld_dataset"
    if not base.exists():
        print(f"ERROR: base directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    splits = ["train", "val", "test"]
    all_landmarks: OrderedDict[str, dict[str, list[float]]] = OrderedDict()
    all_spacings:  OrderedDict[str, dict[str, list[float] | None]] = OrderedDict()

    for split in splits:
        for vol in sorted((base / split).glob("*_volume.nrrd")):
            case_id = vol.stem.replace("_volume", "")  # already <dataset>_sub-XXXX

            # -- image spacing --------------------------------------------------
            try:
                _data, hdr = nrrd.read(str(vol), index_order="C")
                try:
                    img_sp = spacing_from_header(hdr)
                except ValueError:
                    img_sp = None
            except Exception:
                img_sp = None

            # -- annotation spacing -------------------------------------------
            ann_sp_npy = (base / split / f"{case_id}_spacing.npy")
            if ann_sp_npy.exists():
                try:
                    raw = np.load(str(ann_sp_npy))
                    ann_sp = tuple(float(x) for x in raw)
                except Exception:
                    ann_sp = None
            else:
                ann_sp = None

            # -- landmarks ------------------------------------------------------
            lbl_npy = base / split / f"{case_id}_label.npy"
            if not lbl_npy.exists():
                continue
            coords = np.load(str(lbl_npy))
            if coords.ndim != 2 or coords.shape[1] != 3:
                continue

            lm_dict: OrderedDict[str, list[float]] = OrderedDict()
            for idx, (x, y, z) in enumerate(coords):
                lm_dict[f"landmark_{idx}"] = [float(x), float(y), float(z)]

            all_landmarks[case_id] = lm_dict
            all_spacings[case_id] = {
                "image_spacing":      list(img_sp) if img_sp else None,
                "annotation_spacing": list(ann_sp) if ann_sp else None,
            }

    # ----------------------------------------------------------------- outputs
    out_root = Path("/path/to/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset733_MML/")  # keep JSONs inside dataset folder
    (out_root / "all_landmarks_voxel.json").write_text(
        json.dumps(all_landmarks, indent=2))

    # ---- build name→label ordered by first appearance ------------------------
    name_to_label: OrderedDict[str, int] = OrderedDict()
    for lm in all_landmarks.values():           # preserves case insertion order
        for name in lm:                         # preserves landmark index order
            if name not in name_to_label:
                tail = name.split("_")[-1]
                if tail.isdigit():
                    label = int(tail) + 1       # reserve 0 for background
                else:
                    label = len(name_to_label) + 1
                name_to_label[name] = label

    (out_root / "name_to_label.json").write_text(
        json.dumps(name_to_label, indent=2))

    (out_root / "spacing.json").write_text(
        json.dumps(all_spacings, indent=2))

    print("✅  Wrote JSONs to", out_root)


if __name__ == "__main__":
    main()


