#!/usr/bin/env python3
import json
import sys
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import nibabel as nib

def get_nifti_orientation(nifti_file: Path) -> str:
    img = nib.load(str(nifti_file))
    ornt = nib.orientations.io_orientation(img.affine)
    code = nib.orientations.ornt2axcodes(ornt)
    return "".join(code)

def list_cases(images_dir: Path) -> List[str]:
    cases = []
    for p in sorted(images_dir.glob("*.nii")) + sorted(images_dir.glob("*.nii.gz")):
        cases.append(p.stem.replace(".nii", ""))  # handles .nii.gz safely
    return sorted(set(cases))

def parse_mrk_json(mrk_path: Path) -> Dict:
    # Slicer Markups JSON v1.x; expect fields: markups[0].coordinateSystem, controlPoints[*].position
    with mrk_path.open() as f:
        data = json.load(f)
    return data

def extract_points_mm(markup: Dict) -> Tuple[str, np.ndarray]:
    m = markup["markups"][0]
    coord_sys = m.get("coordinateSystem", "LPS")
    pts = np.array([cp["position"] for cp in m.get("controlPoints", [])], dtype=float)
    return coord_sys, pts

def lps_to_ras(xyz_lps: np.ndarray) -> np.ndarray:
    xyz_ras = xyz_lps.copy()
    xyz_ras[..., 0] = -xyz_ras[..., 0]  # L->R
    xyz_ras[..., 1] = -xyz_ras[..., 1]  # P->A
    # S unchanged
    return xyz_ras

def analyze_split(split_name: str, img_dir: Path, lbl_dir: Path, spacing_tol=0.01):
    print(f"\n[SPLIT] {split_name}")
    cases_img = list_cases(img_dir)
    print(f"Images: {len(cases_img)} cases in {img_dir}")  # simple count

    per_case_landmark_counts = Counter()
    case_missing = defaultdict(list)
    spacing_counter = Counter()
    spacing_example = {}
    orient_counter = Counter()
    orient_example = {}

    for case in cases_img:
        img_path = None
        for ext in (".nii.gz", ".nii"):
            p = img_dir / f"{case}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"[WARN] Missing image for case {case}")
            continue

        # scan label folder
        case_lbl_dir = lbl_dir / case
        mrk_files = sorted(case_lbl_dir.glob("*.mrk.json"))
        n_ok = 0
        for mf in mrk_files:
            try:
                mk = parse_mrk_json(mf)
                coord_sys, pts = extract_points_mm(mk)
                # For ‘Line’ markups with two points, count both defined points
                pts = np.asarray(pts, dtype=float)
                n_defined = int((~np.isnan(pts).any(axis=1)).sum())
                n_ok += n_defined
            except Exception as e:
                case_missing[case].append(str(mf.name))
        per_case_landmark_counts[n_ok] += 1

        # spacing and orientation
        try:
            img = nib.load(str(img_path))
            spacing = tuple(np.round(img.header.get_zooms(), 5))
            rounded = tuple(round(v / spacing_tol) * spacing_tol for v in spacing)
            spacing_counter[rounded] += 1
            spacing_example.setdefault(rounded, img_path.name)

            orientation = get_nifti_orientation(img_path)
            orient_counter[orientation] += 1
            orient_example.setdefault(orientation, img_path.name)
        except Exception as e:
            print(f"[WARN] NIfTI read error for {case}: {e}")

    print("Histogram of total defined control points per case:")
    for k, v in sorted(per_case_landmark_counts.items()):
        print(f"  {k}: {v} cases")

    if case_missing:
        print("Cases with unreadable label files:")
        for c, files in case_missing.items():
            print(f"  {c}: {len(files)} files failed")
    else:
        print("No unreadable label files detected")

    print("\nVoxel spacing histogram (rounded):")
    for sp, cnt in sorted(spacing_counter.items()):
        print(f"  {sp}: {cnt} cases")
    print("Spacing examples:")
    for sp, ex in spacing_example.items():
        print(f"  {sp}: {ex}")

    print("\nOrientation codes:")
    for oc, cnt in orient_counter.items():
        print(f"  {oc}: {cnt} cases (e.g., {orient_example[oc]})")

def main():
    root = Path("/path/to/2024_Ertl_nnLandmark/data/DMGLD_dataset/LFC_dataset/")
    analyze_split("train", root / "train", root / "train_label")
    analyze_split("valid", root / "valid", root / "valid_label")

if __name__ == "__main__":
    main()
