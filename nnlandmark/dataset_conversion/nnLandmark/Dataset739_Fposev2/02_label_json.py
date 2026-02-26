#!/usr/bin/env python3
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def case_id_from_filename(fn: Path) -> str:
    name = fn.name
    return name[:-12] if name.endswith(".nii.gz") else print("error")


def main():
    vol_dir_tr = Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/imagesTr")
    vol_dir_ts = Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/imagesTs")
    anno_dir = Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/data/landmark_datasets_Shenzhen/pose_dataset/landmark_anno_v2")

    vols  = list(vol_dir_tr.glob("*.nii.gz")) + list(vol_dir_ts.glob("*.nii.gz"))
    annos = list(anno_dir.glob("*_landmark.json"))

    vol_map  = {case_id_from_filename(v): v for v in vols}
    anno_map = {p.stem.replace("_landmark", ""): p for p in annos}
    cases    = sorted(set(vol_map) & set(anno_map))

    out_labels, out_spacing = {}, {}

    for case in cases:
        data       = json.loads(anno_map[case].read_text())
        anno_spac  = float(data.get("spacing", np.nan))
        landmarks  = data["landmarks"]                 # name → [i,j,k]

        # keep original landmark order from JSON
        out_labels[case] = {n: [float(c) for c in coord] for n, coord in landmarks.items()}
        out_spacing[case] = {
            "image_spacing": None,
            "annotation_spacing": [anno_spac,anno_spac,anno_spac]
        }

    # ---------- ORDER-PRESERVED build of name → label ----------------------
    name_to_label = {}
    for case in cases:                     # case order
        for name in out_labels[case]:      # landmark order inside each file
            if name not in name_to_label:
                tail = name.split("_")[-1]
                label = int(tail) if tail.isdigit() else len(name_to_label) + 1
                name_to_label[name] = label
    # -----------------------------------------------------------------------

    out_root = Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2")
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "all_landmarks_voxel.json").write_text(
        json.dumps(out_labels, indent=2)
    )
    (out_root / "spacing.json").write_text(
        json.dumps(out_spacing, indent=2)
    )
    (out_root / "name_to_label.json").write_text(
        json.dumps(name_to_label, indent=2)
    )

    print(f"✅  wrote landmarks ({len(cases)} cases), spacing and name_to_label JSONs to {out_root}")


if __name__ == "__main__":
    main()

