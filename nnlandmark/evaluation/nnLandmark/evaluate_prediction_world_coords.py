#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate AFIDs predictions by converting voxel-space predictions to world (mm) coordinates
using the affine from the nnLM_raw imagesTs image, and comparing to original AFIDs FCSV.

Assumptions
-----------
- Prediction JSON files are named like: AFIDs-HCP_sub-123925.json
  and contain keys that are numeric labels-as-strings ("1","2",...) with:
    {
      "1": {"coordinates": [x, y, z], ...},
      "2": {"coordinates": [x, y, z], ...},
      ...
    }
  (This mirrors how evaluate_MRE read them in your trainer.)

- The matching images live in imagesTs and are named like:
    AFIDs-HCP_sub-123925_0000.nii.gz
  (the script will try with and without the `_0000` suffix just in case).

- AFIDs ground-truth FCSV lives in:
    /home/a332l/dev/Project_nnLandmark/data/afids-data/data/datasets/AFIDs-*/derivatives/afids_groundtruth/sub-<ID>/anat/sub-<ID>_space-T1w_desc-groundtruth_afids.fcsv

- World coordinates are taken as NIfTI RAS mm (nibabel’s affine). This should match AFIDs.

Usage
-----
python eval_voxel2world.py \
  --pred_dir /path/to/pred_jsons \
  --imagesTs_root /path/to/nnUNet_raw/DatasetXXX/imagesTs \
  --afids_root /home/a332l/dev/Project_nnLandmark/data/afids-data/data/datasets \
  --out_dir /path/to/save/world_jsons_and_summary
"""

import argparse
import json
import math
import os
import re
from glob import glob
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

# ---------------------------- Utilities ----------------------------

def load_json(fp: str):
    with open(fp, "r") as f:
        return json.load(f)

def save_json(obj, fp: str):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)

def case_id_from_fname(fname: str) -> str:
    # "SNSX_sub-C005.json" -> "SNSX_sub-C005"
    return os.path.splitext(os.path.basename(fname))[0]


def guess_image_path(imagesTs_root: str, case_id: str) -> str:
    import glob, os
    # direct candidates under imagesTs_root
    cands = [
        os.path.join(imagesTs_root, f"{case_id}_0000.nii.gz"),
        os.path.join(imagesTs_root, f"{case_id}.nii.gz"),
        os.path.join(imagesTs_root, f"{case_id}_0000.nii"),
        os.path.join(imagesTs_root, f"{case_id}.nii"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c

    # recursive search one level below (handles layouts like */imagesTs/<files>)
    patterns = [
        os.path.join(imagesTs_root, "**", f"{case_id}_0000.nii.gz"),
        os.path.join(imagesTs_root, "**", f"{case_id}.nii.gz"),
        os.path.join(imagesTs_root, "**", f"{case_id}_0000.nii"),
        os.path.join(imagesTs_root, "**", f"{case_id}.nii"),
    ]
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return sorted(hits)[0]

    raise FileNotFoundError(f"Could not find image for case '{case_id}' under '{imagesTs_root}'")


def voxel_to_world(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """
    ijk is in voxel index order matching NIfTI array axes (i, j, k) = (x, y, z) index.
    Returns RAS mm coordinates.
    """
    return nib.affines.apply_affine(affine, ijk)

def find_afids_fcsv(afids_root: str, case_id: str) -> str:
    """
    case_id examples: "AFIDs-HCP_sub-103111", "SNSX_sub-C005"
    Looks under:
      {afids_root}/{subset}/derivatives/afids_groundtruth/sub-<SID>/anat/*.fcsv
    """
    import glob, os, re
    subset = case_id.split("_", 1)[0]  # "AFIDs-HCP" or "SNSX" etc.
    m = re.search(r"sub-([A-Za-z0-9]+)", case_id)   # accept alphanumerics
    if not m:
        raise RuntimeError(f"Could not extract subject ID from case_id '{case_id}'")
    sid = m.group(1)

    # canonical candidate
    cand = os.path.join(
        afids_root, subset, "derivatives", "afids_groundtruth",
        f"sub-{sid}", "anat", f"sub-{sid}_space-T1w_desc-groundtruth_afids.fcsv"
    )
    if os.path.isfile(cand):
        return cand

    # fallback: any .fcsv in that anat dir, prefer names containing "groundtruth"
    pat = os.path.join(
        afids_root, subset, "derivatives", "afids_groundtruth",
        f"sub-{sid}", "anat", "*.fcsv"
    )
    hits = sorted(glob.glob(pat))
    if hits:
        for h in hits:
            if "groundtruth" in os.path.basename(h).lower():
                return h
        return hits[0]

    raise FileNotFoundError(f"FCSV not found. Tried:\n  {cand}\n  and pattern:\n  {pat}")

def parse_fcsv(fp: str) -> Dict[str, List[float]]:
    """
    Parse a 3D Slicer Markups FCSV and return:
        {'landmark_<n>': [x, y, z], ...}
    If no integer can be parsed from the label, keep the raw label as the key.
    """
    coords = {}
    with open(fp, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Try to read header mapping
    col_idx = {"x": None, "y": None, "z": None, "label": None}
    for ln in lines:
        if ln.startswith("# columns"):
            cols = ln.split("=", 1)[1].strip()
            names = [c.strip() for c in cols.split(",")]
            for k in ["x", "y", "z", "label"]:
                if k in names:
                    col_idx[k] = names.index(k)
            break

    # Defaults if header line is absent
    if col_idx["x"] is None:
        # Slicer classic: id(0),label(1),desc(2),assocNodeID(3),x(4),y(5),z(6),...
        col_idx = {"label": 1, "x": 4, "y": 5, "z": 6}

    for ln in lines:
        if ln.startswith("#"):
            continue
        parts = [p.strip() for p in ln.split(",")]
        try:
            label_str = parts[col_idx["label"]]
            x = float(parts[col_idx["x"]]); y = float(parts[col_idx["y"]]); z = float(parts[col_idx["z"]])
        except Exception:
            continue

        # Extract the FIRST integer appearing in the label (e.g., "AFID01", "landmark_1", "1")
        m = re.search(r"(\d+)", label_str)
        if m:
            n = int(m.group(1))
            key = f"landmark_{n}"
        else:
            key = label_str  # fallback if label has no digits
        coords[key] = [x, y, z]
    return coords

# ---------------------------- Core ----------------------------

def convert_pred_voxel_to_world_for_case(pred_json_path: str,
                                         imagesTs_root: str,
                                         out_world_json_dir: str) -> Tuple[str, Dict[str, List[float]]]:
    """
    Returns (case_id, world_coords_dict)
    Saves world coords JSON as {out_world_json_dir}/{case_id}_world.json
    """
    case_id = case_id_from_fname(pred_json_path)
    img_path = guess_image_path(imagesTs_root, case_id)
    img = nib.load(img_path)
    aff = img.affine

    pred = load_json(pred_json_path)
    # Build a dict of landmark_{n} -> [X,Y,Z] mm
    world = {}
    for k, v in pred.items():
        # keys are numeric labels-as-strings in predictions (e.g., "1", "2", ...)
        try:
            n = int(k)
            key = f"landmark_{n}"
        except Exception:
            key = k  # fallback
        ijk = np.array(v["coordinates"], dtype=float)  # voxel indices
        ras = voxel_to_world(aff, ijk)
        world[key] = [float(ras[0]), float(ras[1]), float(ras[2])]

    out_fp = os.path.join(out_world_json_dir, f"{case_id}_world.json")
    save_json(world, out_fp)
    return case_id, world

def compare_world_to_fcsv(world_coords: Dict[str, List[float]], fcsv_coords: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Return per-landmark Euclidean errors in mm for any landmark present in BOTH dicts.
    """
    errs = {}
    common = sorted(set(world_coords.keys()) & set(fcsv_coords.keys()),
                    key=lambda s: (s.startswith("landmark_"), int(re.search(r"\d+", s).group(0)) if re.search(r"\d+", s) else 1e9, s))
    for key in common:
        p = np.array(world_coords[key], dtype=float)
        g = np.array(fcsv_coords[key], dtype=float)
        errs[key] = float(np.linalg.norm(p - g))
    return errs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default='/home/a332l/dev/Project_nnLandmark/evaluation/Dataset732_Afids/nnLandmark_v1/prediction/',
                    help="Folder with prediction JSONs (voxel coords).")
    ap.add_argument("--imagesTs_root", default='/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset732_Afids/imagesTs/',
                    help="nnLM_raw Dataset.../imagesTs directory with the original NIfTI files.")
    ap.add_argument("--afids_root", default='/home/a332l/dev/Project_nnLandmark/data/afids-data/data/datasets/',
                    help="AFIDs ground truth root, e.g., /home/.../AFIDs-HCP/derivatives/afids_groundtruth")
    ap.add_argument("--out_dir", default='/home/a332l/dev/Project_nnLandmark/evaluation/Dataset732_Afids/nnLandmark_v1/',
                    help="Output directory for *_world.json and summary.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    world_json_dir = os.path.join(args.out_dir, "world_jsons")
    os.makedirs(world_json_dir, exist_ok=True)

    pred_jsons = []
    for f in os.listdir(args.pred_dir):
        if not f.endswith(".json"):
            continue
        if f in {"summary.json", "dataset.json", "spacing.json", "name_to_label.json",
                "all_landmarks_voxel.json", "landmark_coordinates.json"}:
            continue
        if "sub-" not in f:
            # skip files like "dataset.json" that would yield case_id="dataset"
            continue
        pred_jsons.append(os.path.join(args.pred_dir, f))
    pred_jsons.sort()
    if not pred_jsons:
        raise RuntimeError(f"No prediction JSONs (with 'sub-') found in {args.pred_dir}")

    detailed = {}
    all_errs_by_landmark: Dict[str, List[float]] = {}

    for pj in pred_jsons:
        case_id, world_coords = convert_pred_voxel_to_world_for_case(pj, args.imagesTs_root, world_json_dir)
        try:
            fcsv_path = find_afids_fcsv(args.afids_root, case_id)
            fcsv_coords = parse_fcsv(fcsv_path)
        except Exception as e:
            print(f"[WARN] Skipping comparison for {case_id}: {e}")
            continue

        errs = compare_world_to_fcsv(world_coords, fcsv_coords)
        detailed[case_id] = errs
        # aggregate
        for k, v in errs.items():
            all_errs_by_landmark.setdefault(k, []).append(v)

    # Summaries
    mre_by_landmark = {k: float(np.mean(v)) for k, v in sorted(all_errs_by_landmark.items(),
                                                               key=lambda kv: (kv[0].startswith("landmark_"),
                                                                               int(re.search(r"\d+", kv[0]).group(0)) if re.search(r"\d+", kv[0]) else 1e9,
                                                                               kv[0]))}
    overall_mre = float(np.mean(list(mre_by_landmark.values()))) if mre_by_landmark else float("nan")

    summary = {
        "MRE_mm": overall_mre,
        "MRE_by_landmark_mm": mre_by_landmark,
        "detailed_errors_mm": detailed,
        "notes": {
            "coords": "World coords are RAS (mm) via NIfTI affine.",
            "imagesTs_root": args.imagesTs_root,
            "afids_root": args.afids_root
        }
    }
    save_json(summary, os.path.join(args.out_dir, "summary_world.json"))
    print(f"Done. Saved world-coord JSONs under:\n  {world_json_dir}\nSummary:\n  {os.path.join(args.out_dir, 'summary_world.json')}")

if __name__ == "__main__":
    main()
