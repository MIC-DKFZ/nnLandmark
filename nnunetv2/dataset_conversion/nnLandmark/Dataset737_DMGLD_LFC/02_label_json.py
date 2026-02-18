#!/usr/bin/env python3
from __future__ import annotations  # optional; keeps annotations as strings
import json
import argparse
import re
from pathlib import Path
from collections import OrderedDict
from typing import Optional, List, Dict, Tuple

import numpy as np
import nibabel as nib

# Optional natural-sort helper for .mrk.json files (uses first integer in name)
FN_NUM_RE = re.compile(r"(\d+)")

def lps_to_ras(xyz_lps: np.ndarray) -> np.ndarray:
    xyz_ras = xyz_lps.copy()
    xyz_ras[..., 0] = -xyz_ras[..., 0]
    xyz_ras[..., 1] = -xyz_ras[..., 1]
    return xyz_ras

def inv_affine(aff: np.ndarray) -> np.ndarray:
    return np.linalg.inv(aff)

def worldmm_to_voxel(xyz_mm_ras: np.ndarray, affine: np.ndarray) -> np.ndarray:
    invA = inv_affine(affine)
    ones = np.ones((xyz_mm_ras.shape[0], 1), dtype=float)
    homog = np.concatenate([xyz_mm_ras, ones], axis=1)
    ijk = (invA @ homog.T).T[:, :3]
    return ijk

def parse_mrk_json(mrk_path: Path) -> Dict:
    with mrk_path.open() as f:
        return json.load(f)

def extract_coord_sys_and_points(markup: Dict) -> Tuple[str, List[Dict]]:
    m = markup["markups"][0]
    coord_sys = m.get("coordinateSystem", "LPS")
    cps = m.get("controlPoints", [])
    return coord_sys, cps

def image_for_case(img_dir: Path, case: str) -> Optional[Path]:
    for ext in (".nii.gz", ".nii"):
        p = img_dir / f"{case}{ext}"
        if p.exists():
            return p
    return None

def cases_in_dir(img_dir: Path) -> List[str]:
    names = set()
    for p in sorted(img_dir.glob("*.nii")) + sorted(img_dir.glob("*.nii.gz")):
        stem = p.name
        if stem.endswith(".nii.gz"):
            case = stem[:-7]
        elif stem.endswith(".nii"):
            case = stem[:-4]
        else:
            continue
        names.add(case)
    return sorted(names)

def nat_key_mrk(p: Path) -> Tuple[int, str]:
    """Natural sort by first integer in filename, fallback to name."""
    stem = p.stem.replace(".mrk", "")
    m = FN_NUM_RE.search(stem)
    return (int(m.group(1)) if m else 10**9, stem)

def collect_split_many_points(src_img_dir: Path, src_lbl_dir: Path) -> Tuple[OrderedDict, OrderedDict, List[str]]:
    """
    For each case:
      - Load image and all *.mrk.json in case folder.
      - Sort files deterministically (natural sort on first integer in name).
      - For each file index (measurement_idx starting at 1) and each control point index (cp_idx starting at 1),
        emit name = landmark_{measurement_idx}_{cp_idx}.
    """
    all_labels: OrderedDict[str, Dict[str, List[float]]] = OrderedDict()
    all_spacing: OrderedDict[str, Dict[str, Optional[List[float]]]] = OrderedDict()
    all_names_set = set()

    cases = cases_in_dir(src_img_dir)
    for case in cases:
        img_path = image_for_case(src_img_dir, case)
        if img_path is None:
            print(f"[WARN] image missing for case {case}")
            continue

        try:
            img = nib.load(str(img_path))
        except Exception as e:
            print(f"[WARN] failed to load image for {case}: {e}")
            continue
        affine = img.affine
        spacing = list(map(float, img.header.get_zooms()))

        case_dir = src_lbl_dir / case
        if not case_dir.is_dir():
            print(f"[WARN] label folder missing: {case_dir}")
            continue

        lm_dict: Dict[str, List[float]] = OrderedDict()
        mrk_files = sorted(case_dir.glob("*.mrk.json"), key=nat_key_mrk)

        for meas_idx, mf in enumerate(mrk_files, start=1):
            try:
                mk = parse_mrk_json(mf)
                coord_sys, cps = extract_coord_sys_and_points(mk)

                for cp_idx, cp in enumerate(cps, start=1):
                    pos = np.asarray(cp.get("position", [np.nan, np.nan, np.nan]), dtype=float).reshape(1, 3)
                    if np.isnan(pos).any():
                        print(f"[WARN] NaN position for {case} {mf.name} cp#{cp_idx}, skipping")
                        continue

                    # Convert LPS mm -> RAS mm if needed
                    if coord_sys.upper() == "LPS":
                        pos = lps_to_ras(pos)

                    # World (RAS mm) -> voxel ijk
                    ijk = worldmm_to_voxel(pos, affine)[0]

                    # Name by enumeration only (ignore filename labels entirely)
                    name = f"landmark_{meas_idx}_{cp_idx}"
                    lm_dict[name] = [float(ijk[0]), float(ijk[1]), float(ijk[2])]
                    all_names_set.add(name)

            except Exception as e:
                print(f("[WARN] error parsing {}: {}").format(mf, e))

        if not lm_dict:
            print(f"[WARN] no landmarks collected for {case}")
            continue

        all_labels[case] = lm_dict
        all_spacing[case] = {"image_spacing": spacing, "annotation_spacing": None}

    # Sort names by (measurement_idx, cp_idx)
    def name_key(s: str) -> Tuple[int, int]:
        m = re.match(r"^landmark_(\d+)_(\d+)$", s)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (10**9, 10**9)

    return all_labels, all_spacing, sorted(all_names_set, key=name_key)

def main():
    ap = argparse.ArgumentParser(
        description="Convert multi-point Slicer .mrk.json into nnLandmark JSONs; each control point becomes a distinct landmark named by enumeration: landmark_{measurement_idx}_{cp_idx}"
    )
    ap.add_argument("--src", required=True, help="Source root with train/, valid/, train_label/, valid_label/")
    ap.add_argument("--dst", required=True, help="Destination root to write JSONs (dataset folder)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    #tr_labels, tr_spacing, tr_names = collect_split_many_points(src / "train", src / "train_label")
    #ts_labels, ts_spacing, ts_names = collect_split_many_points(src / "valid", src / "valid_label")
    ts_labels, ts_spacing, ts_names = collect_split_many_points(src / "feta21", src / "feta21_label")

    # Build contiguous name_to_label in enumerated order
    #all_names_sorted = sorted(set(tr_names).union(ts_names),
    all_names_sorted = sorted(ts_names,
                              key=lambda s: (int(re.match(r"^landmark_(\d+)_(\d+)$", s).group(1)),
                                             int(re.match(r"^landmark_(\d+)_(\d+)$", s).group(2)))
                              if re.match(r"^landmark_(\d+)_(\d+)$", s) else (10**9, 10**9))
    name_to_label: Dict[str, int] = {name: idx for idx, name in enumerate(all_names_sorted, start=1)}

    #(dst / "all_landmarks_voxel_train.json").write_text(json.dumps(tr_labels, indent=2))
    #(dst / "all_landmarks_voxel_test.json").write_text(json.dumps(ts_labels, indent=2))
    #(dst / "spacing_train.json").write_text(json.dumps(tr_spacing, indent=2))
    (dst / "spacing_test.json").write_text(json.dumps(ts_spacing, indent=2))
    (dst / "name_to_label.json").write_text(json.dumps(name_to_label, indent=2))
    (dst / "all_landmarks_voxel_test.json").write_text(json.dumps(ts_labels, indent=2))

    print("Wrote:")
    print("  all_landmarks_voxel_train.json")
    print("  all_landmarks_voxel_test.json")
    print("  spacing_train.json")
    print("  spacing_test.json")
    print("  name_to_label.json")

if __name__ == "__main__":
    main()

