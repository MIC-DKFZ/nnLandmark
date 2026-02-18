#!/usr/bin/env python3
"""
FeTA Dataset Analysis Script

- Extracts landmarks from segmentation maps.
- Counts the number of valid landmarks (labels with exactly 2 points) per case.
- Generates a histogram of the number of landmarks per case.
- Outputs a dataset overview JSON file and prints a concise summary.
"""

from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import nibabel as nib


def _extract_landmarks_from_meas_corrected(meas_path: Path) -> Dict[int, int]:
    """
    Extract landmarks from the segmentation map and count the number of points for each label.

    Args:
        meas_path (Path): Path to the segmentation map.

    Returns:
        Dict[int, int]: Dictionary of label -> number of points.
    """
    if nib is None:
        raise RuntimeError("nibabel required (pip install nibabel)")
    try:
        img = nib.load(str(meas_path))
        arr = np.asarray(img.dataobj)
    except Exception:
        return {}

    landmarks = {}
    for label in np.unique(arr):
        if label == 0:  # Skip background
            continue
        coords = np.argwhere(arr == label)
        num_points = coords.shape[0]
        if num_points != 2:
            print(f"[WARN] Label {label} in {meas_path} does not have exactly 2 points (found {num_points}).")
        landmarks[label] = num_points
    return landmarks


def _get_nifti_shape_spacing(p: Path) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Get the shape and spacing of a NIfTI image.

    Args:
        p (Path): Path to the NIfTI image.

    Returns:
        Tuple[Tuple[int, int, int], Tuple[float, float, float]]: Shape and spacing of the image.
    """
    img = nib.load(str(p))
    shape = tuple(img.shape)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return shape, spacing


def scan_dataset_corrected(root: Path) -> Dict[str, Any]:
    """
    Scan the dataset and generate an overview.

    Args:
        root (Path): Root directory of the dataset.

    Returns:
        Dict[str, Any]: Dataset overview.
    """
    root = Path(root)
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

    total_cases = len(case_dirs)
    missing_anat = []
    missing_meas = []
    shape_counter = Counter()
    spacing_list: List[Tuple[float, float, float]] = []
    landmarks_per_case = []
    label_case_presence = Counter()
    label_voxel_counts = Counter()
    per_case_details = {}

    for case_dir in case_dirs:
        cid = case_dir.name
        print("[INFO] Scanning case:", cid)

        anat_candidates = list(case_dir.glob("**/anat/*_T2w*.nii*"))
        anat_path = anat_candidates[0] if anat_candidates else None
        if anat_path:
            shape, spacing = _get_nifti_shape_spacing(anat_path)
            if shape:
                shape_counter[tuple(shape[:3])] += 1
            if spacing:
                spacing_list.append(spacing)
        else:
            missing_anat.append(cid)
            shape = None
            spacing = None

        meas_candidates = list(root.joinpath("derivatives", "biometry", cid).glob("anat/*_meas*.nii*"))
        meas_path = meas_candidates[0] if meas_candidates else None
        if not meas_path:
            missing_meas.append(cid)
            landmarks = {}
        else:
            landmarks = _extract_landmarks_from_meas_corrected(meas_path)

        # Count the number of valid landmarks (labels with exactly 2 points)
        n_landmarks = sum(1 for lbl, count in landmarks.items() if count == 2)
        landmarks_per_case.append(n_landmarks)
        for lbl, count in landmarks.items():
            if count == 2:  # Only count valid landmarks
                label_case_presence[lbl] += 1
                label_voxel_counts[lbl] += count

        per_case_details[cid] = {
            "anat": str(anat_path.name) if anat_path else None,
            "meas": str(meas_path.name) if meas_path else None,
            "shape": list(shape) if shape else None,
            "spacing": list(spacing) if spacing else None,
            "n_landmarks": n_landmarks,
        }

    # Build histograms/statistics
    spacing_arr = np.array(spacing_list) if spacing_list else np.zeros((0, 3))
    spacing_stats = {}
    if spacing_arr.size:
        spacing_stats = {
            "count_cases_with_spacing": int(spacing_arr.shape[0]),
            "spacing_mean": [float(np.mean(spacing_arr[:, i])) for i in range(3)],
            "spacing_std": [float(np.std(spacing_arr[:, i])) for i in range(3)],
            "unique_spacings": sorted({tuple([float(round(x, 6)) for x in t]) for t in spacing_list}),
        }

    shape_hist = {str(k): int(v) for k, v in shape_counter.items()}
    landmarks_counter = Counter(landmarks_per_case)
    landmarks_stats = {
        "per_case_counts": landmarks_per_case,
        "histogram": dict(landmarks_counter),
        "min": int(min(landmarks_per_case)) if landmarks_per_case else 0,
        "max": int(max(landmarks_per_case)) if landmarks_per_case else 0,
        "mean": float(np.mean(landmarks_per_case)) if landmarks_per_case else 0.0,
        "median": float(np.median(landmarks_per_case)) if landmarks_per_case else 0.0,
    }

    label_presence = {str(k): {"cases": int(v), "voxels_total": int(label_voxel_counts.get(k, 0))} for k, v in label_case_presence.items()}

    overview = {
        "root": str(root),
        "total_cases": total_cases,
        "cases_with_missing_anat": missing_anat,
        "cases_with_missing_meas": missing_meas,
        "n_cases_missing_anat": len(missing_anat),
        "n_cases_missing_meas": len(missing_meas),
        "shape_histogram": shape_hist,
        "spacing_stats": spacing_stats,
        "landmarks_stats": landmarks_stats,
        "label_presence": label_presence,
        "per_case_details_sample": dict(list(per_case_details.items())[:20]),
    }
    return overview


def main():
    root = Path("/path/to/2024_Ertl_nnLandmark/data/feta_2.4")
    dataset_dir = Path("/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/")
    

    overview = scan_dataset_corrected(root)
    out_path = dataset_dir / "dataset_overview.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(overview, f, indent=2)

    # Concise console print
    print(f"Dataset root: {overview['root']}")
    print(f"Total cases: {overview['total_cases']}")
    print(f"Cases missing anat: {overview['n_cases_missing_anat']}, missing meas: {overview['n_cases_missing_meas']}")
    ss = overview["spacing_stats"]
    if ss:
        print("Spacing (mean):", ss["spacing_mean"], "std:", ss["spacing_std"])
        print("Unique spacings (sample):", ss["unique_spacings"][:8])
    print("Image shape distribution (top):", {k: overview["shape_histogram"][k] for k in list(overview["shape_histogram"])[:5]})
    ls = overview["landmarks_stats"]
    print("Landmarks per case: min/max/mean/median:", ls["min"], ls["max"], round(ls["mean"], 2), round(ls["median"], 2))
    print("Label presence (counts of cases):")
    for lbl, info in sorted(overview["label_presence"].items(), key=lambda x: int(x[0])):
        print(f"  label {lbl}: cases={info['cases']}, voxels_total={info['voxels_total']}")
    print(f"Wrote overview to: {out_path}")


if __name__ == "__main__":
    main()