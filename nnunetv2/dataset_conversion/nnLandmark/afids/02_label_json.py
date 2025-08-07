#!/usr/bin/env python3
"""
Extract AFIDs landmarks and write two JSON files:

    afids_all_landmarks.json   # case → { landmark_<n>: [i,j,k] }
    afids_name_to_label.json   # landmark_<n> → integer label (1-32)

Only landmarks 1-32 are kept; any others are ignored with a warning.
"""

import json
import re
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np


# -------------------------------------------------------------------------#
# helpers                                                                  #
# -------------------------------------------------------------------------#
NUMBER_RE   = re.compile(r"^(\d+)$")
VTK_RE      = re.compile(r"^vtkMRMLMarkupsFiducialNode_(\d+)$")


def normalise_name(raw: str) -> str | None:
    """Return canonical landmark name or *None* if the row should be skipped."""
    m = NUMBER_RE.match(raw) or VTK_RE.match(raw)
    if not m:
        return None                       # unknown naming – skip the landmark
    idx = int(m.group(1))
    if not (1 <= idx <= 32):              # outside supported range
        return None
    return f"landmark_{idx}"


def parse_fcsv(fcsv_path: Path) -> OrderedDict[str, list[float]]:
    """
    Read a 3-D Slicer .fcsv and return an OrderedDict
        landmark_<n>  ->  [x, y, z] (world-mm)
    Only landmarks 1–32 are kept.
    """
    lm: OrderedDict[str, list[float]] = OrderedDict()
    with fcsv_path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")

            raw_name = parts[0]
            name = normalise_name(raw_name)
            if name is None:
                continue      # ignore non-AFIDs points

            try:
                x, y, z = map(float, parts[1:4])
                lm[name] = [x, y, z]
            except ValueError:
                print(f"[WARN] bad row in {fcsv_path.name}: {line.strip()}")
    return lm


def mm_to_voxel(coords_mm: list[list[float]], affine: np.ndarray) -> list[list[float]]:
    """Convert world-mm coords → voxel (i,j,k) using the affine inverse."""
    inv_affine = np.linalg.inv(affine)
    return [
        (inv_affine @ np.array([x, y, z, 1.0]))[:3].tolist()
        for x, y, z in coords_mm
    ]


# -------------------------------------------------------------------------#
# main                                                                     #
# -------------------------------------------------------------------------#
def main() -> None:
    root = Path("/home/a332l/dev/Project_SoftDiceLoss/data/afids-data/data/datasets")
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    all_labels:  dict[str, dict[str, list[float]]] = {}
    all_spacing: dict[str, dict[str, list[float] | None]] = {}
    name_to_label = {f"landmark_{i}": i for i in range(1, 33)}

    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        fcsvs = sorted(ds.glob("derivatives/afids_groundtruth/sub-*/anat/*.fcsv"))
        for fcsv in fcsvs:
            sub_id   = fcsv.parents[1].name        # 'sub-XXXX'
            case_key = f"{ds.name}_{sub_id}"

            lm_mm = parse_fcsv(fcsv)
            if not lm_mm:
                print(f"[WARN] No usable landmarks in {fcsv}")
                continue

            anat_dir = ds / sub_id / "anat"
            nifti_files = sorted(anat_dir.glob("*.nii*"))
            if not nifti_files:
                print(f"[WARN] No NIfTI for {case_key}")
                continue
            nifti = nib.load(str(nifti_files[0]))

            lm_vox = mm_to_voxel(list(lm_mm.values()), nifti.affine)
            all_labels[case_key] = OrderedDict(
                (name, coord) for (name, _), coord in zip(lm_mm.items(), lm_vox)
            )
            all_spacing[case_key] = {
                "image_spacing":      list(map(float, nifti.header.get_zooms())),
                "annotation_spacing": None,
            }

    out_root = Path("/home/a332l/dev/Project_SoftDiceLoss/data/afids-data")
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "all_landmarks_voxel.json").write_text(
        json.dumps(all_labels, indent=2)
    )
    (out_root / "spacing.json").write_text(
        json.dumps(all_spacing, indent=2)
    )
    (out_root / "name_to_label.json").write_text(
        json.dumps(name_to_label, indent=2)
    )
    print("✅  Wrote:")
    print("   • afids_all_landmarks.json")
    print("   • afids_spacing.json")
    print("   • afids_name_to_label.json")


if __name__ == "__main__":
    main()
