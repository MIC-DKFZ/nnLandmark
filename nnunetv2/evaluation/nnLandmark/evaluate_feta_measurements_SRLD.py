#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

def load_json(file_path: Path) -> Dict:
    """Load a JSON file."""
    with file_path.open() as f:
        return json.load(f)

def calculate_euclidean_distance(coord1: Tuple[float, float, float], coord2: Tuple[float, float, float]) -> float:
    """Calculate the Euclidean distance between two 3D coordinates."""
    return float(np.sqrt(np.sum((np.array(coord1) - np.array(coord2)) ** 2)))

def extract_spacing_from_entry(entry) -> Tuple[float, float, float]:
    """
    Accepts either:
      - [sx, sy, sz]
      - {"image_spacing": [...]} or {"annotation_spacing": [...]} or {"spacing": [...]}
    Returns tuple of floats (sx, sy, sz)
    """
    if entry is None:
        raise KeyError("Spacing entry is None")
    if isinstance(entry, (list, tuple)) and len(entry) == 3:
        return float(entry[0]), float(entry[1]), float(entry[2])
    if isinstance(entry, dict):
        if "image_spacing" in entry and entry["image_spacing"] is not None:
            vals = entry["image_spacing"]
            return float(vals[0]), float(vals[1]), float(vals[2])
        if "annotation_spacing" in entry and entry["annotation_spacing"] is not None:
            vals = entry["annotation_spacing"]
            return float(vals[0]), float(vals[1]), float(vals[2])
        if "spacing" in entry and entry["spacing"] is not None:
            vals = entry["spacing"]
            return float(vals[0]), float(vals[1]), float(vals[2])
    raise ValueError(f"Unrecognized spacing entry: {entry}")

def build_predicted_points_from_file(predicted_data: Dict) -> Tuple[Dict[str, Tuple[float,float,float]], Optional[Tuple[float,float,float]]]:
    """
    Normalize predicted data file into:
      - predicted_points: { 'landmark_1_1': [x,y,z], ... }
      - pred_spacing: tuple or None
    Supports:
      - {'predicted_points': {...}, 'spacing': [...]}
      - {'spacing': [...], 'landmark_1_1': {'coordinates': [...], 'likelihood': ...}, ...}
      - aggregated file not handled here (handled separately)
    """
    # Case: old style with predicted_points key
    if "predicted_points" in predicted_data:
        pred_pts = predicted_data["predicted_points"]
        pred_spacing = None
        if "spacing" in predicted_data:
            pred_spacing = extract_spacing_from_entry(predicted_data["spacing"])
        return pred_pts, pred_spacing

    # Case: per-case top-level spacing + landmark entries like 'landmark_1_1': { 'coordinates': [...] }
    if "spacing" in predicted_data and any(k.startswith("landmark_") for k in predicted_data.keys()):
        pred_spacing = extract_spacing_from_entry(predicted_data.get("spacing"))
        pred_pts = {}
        for k, v in predicted_data.items():
            if not k.startswith("landmark_"):
                continue
            if isinstance(v, dict) and "coordinates" in v:
                pred_pts[k] = v["coordinates"]
            elif isinstance(v, (list, tuple)) and len(v) == 3:
                pred_pts[k] = v
        return pred_pts, pred_spacing

    # Otherwise unknown structure
    # Try to interpret simple mapping landmark -> list as well (defensive)
    pred_pts = {}
    for k, v in predicted_data.items():
        if k.startswith("landmark_") and isinstance(v, (list, tuple)) and len(v) == 3:
            pred_pts[k] = v
    if pred_pts:
        # no spacing in this predicted_data
        return pred_pts, None

    raise ValueError("Unrecognized predicted JSON structure")

def evaluate_case_from_dict(predicted_points: Dict[str, Tuple[float,float,float]],
                            pred_spacing: Optional[Tuple[float,float,float]],
                            ground_truth: Dict[str, Tuple[float,float,float]],
                            gt_spacing_entry,
                            max_landmark_index: int = 6) -> Dict:
    """
    Compute measurements for one case given normalized predicted_points dict and ground truth dict.
    Returns dict with ground_truth_measurements_mm, predicted_measurements_mm, errors_mm
    """
    gt_sx, gt_sy, gt_sz = extract_spacing_from_entry(gt_spacing_entry)
    # If prediction spacing provided use it, otherwise fall back to gt spacing
    if pred_spacing is not None:
        pred_sx, pred_sy, pred_sz = pred_spacing
    else:
        pred_sx, pred_sy, pred_sz = gt_sx, gt_sy, gt_sz

    case_results = {"ground_truth_measurements_mm": {}, "predicted_measurements_mm": {}, "errors_mm": {}}

    for i in range(1, max_landmark_index + 1):
        name1 = f"landmark_{i}_1"
        name2 = f"landmark_{i}_2"
        if name1 in ground_truth and name2 in ground_truth:
            gt_coords1 = ground_truth[name1]
            gt_coords2 = ground_truth[name2]
            gt_dist_vox = calculate_euclidean_distance(gt_coords1, gt_coords2)
            gt_dist_mm = gt_dist_vox * np.sqrt(gt_sx**2 + gt_sy**2 + gt_sz**2)
            case_results["ground_truth_measurements_mm"][f"{name1}-{name2}"] = float(np.round(gt_dist_mm, 6))

            if name1 in predicted_points and name2 in predicted_points:
                pred_coords1 = predicted_points[name1]
                pred_coords2 = predicted_points[name2]
                pred_dist_vox = calculate_euclidean_distance(pred_coords1, pred_coords2)
                pred_dist_mm = pred_dist_vox * np.sqrt(pred_sx**2 + pred_sy**2 + pred_sz**2)
                case_results["predicted_measurements_mm"][f"{name1}-{name2}"] = float(np.round(pred_dist_mm, 6))

                error_mm = abs(gt_dist_mm - pred_dist_mm)
                case_results["errors_mm"][f"{name1}-{name2}"] = float(np.round(error_mm, 6))
    return case_results

def evaluate_measurements(results_dir: Path, ground_truth_path: Path, spacing_path: Path, output_path: Path, max_landmark_index: int = 6):
    """
    Evaluate distances between paired landmarks and compare measurements in mm.
    Accepts:
      - per-case predicted json files (two variants)
      - aggregated prediction_all_landmark_voxel.json
    """
    # Load ground truth and spacing
    ground_truth = load_json(ground_truth_path)
    gt_spacing_map = load_json(spacing_path)

    detailed_results = {}

    results_dir = Path(results_dir)

    # 1) If there is an aggregated prediction_all_landmark_voxel.json process it first
    agg_file = results_dir / "prediction_all_landmark_voxel.json"
    if agg_file.exists():
        agg_preds = load_json(agg_file)
        for case_id, pred_map in agg_preds.items():
            if case_id not in ground_truth:
                print(f"[WARN] case {case_id} in aggregated predictions not found in GT, skipping")
                continue
            if case_id not in gt_spacing_map:
                print(f"[WARN] spacing for case {case_id} not found in spacing.json, skipping")
                continue
            # pred_map: { landmark_name: [x,y,z], ... }
            case_res = evaluate_case_from_dict(predicted_points=pred_map,
                                              pred_spacing=None,  # use GT spacing
                                              ground_truth=ground_truth[case_id],
                                              gt_spacing_entry=gt_spacing_map[case_id],
                                              max_landmark_index=max_landmark_index)
            detailed_results[case_id] = case_res

    # 2) Process per-case predicted files in results_dir
    for predicted_file in sorted(results_dir.glob("*.json")):
        if predicted_file.name in ("prediction_all_landmark_voxel.json", output_path.name):
            continue
        try:
            pd = load_json(predicted_file)
        except Exception as e:
            print(f"[WARN] failed to load {predicted_file}: {e}")
            continue
        # derive case id
        case_id = predicted_file.stem.split("_")[0]
        if case_id not in ground_truth:
            print(f"[WARN] Case '{case_id}' not found in ground truth. Skipping.")
            continue
        if case_id not in gt_spacing_map:
            print(f"[WARN] Spacing for case '{case_id}' not found in ground truth spacing. Skipping.")
            continue

        try:
            pred_pts, pred_spacing = build_predicted_points_from_file(pd)
        except ValueError:
            # try another fallback: if file is single-case format like 1330.json (top-level landmarks + spacing),
            # build_predicted_points_from_file should have handled it; if it didn't, skip
            print(f"[WARN] Unrecognized predicted file structure for {predicted_file}, skipping")
            continue

        case_res = evaluate_case_from_dict(predicted_points=pred_pts,
                                          pred_spacing=pred_spacing,
                                          ground_truth=ground_truth[case_id],
                                          gt_spacing_entry=gt_spacing_map[case_id],
                                          max_landmark_index=max_landmark_index)
        detailed_results[case_id] = case_res

    # Aggregate errors across all cases/pairs
    # collect errors by measurement class
    errors_by_measurement: Dict[str, list] = {}
    for case_res in detailed_results.values():
        for meas_name, err in case_res.get("errors_mm", {}).items():
            errors_by_measurement.setdefault(meas_name, []).append(err)

    # Flatten all errors for global stats
    all_errors = [e for v in errors_by_measurement.values() for e in v]

    mean_error_mm = float(np.round(np.mean(all_errors), 6)) if all_errors else 0.0
    stddev_error_mm = float(np.round(np.std(all_errors), 6)) if all_errors else 0.0

    # per-measurement statistics (mean, std, count)
    mean_error_by_measurement = {}
    stddev_by_measurement = {}
    count_by_measurement = {}
    for means_name, errs in sorted(errors_by_measurement.items()):
        errs_arr = np.array(errs, dtype=float)
        mean_error_by_measurement[means_name] = float(np.round(np.mean(errs_arr), 6))
        stddev_by_measurement[means_name] = float(np.round(np.std(errs_arr), 6))
        count_by_measurement[means_name] = int(len(errs_arr))
    
    class_means = np.array(list(mean_error_by_measurement.values()), dtype=float)
    macro_mean = float(class_means.mean()) if class_means.size else None
    macro_std = float(class_means.std(ddof=0)) if class_means.size else None 

    results_out = {
        "mean_error_mm": mean_error_mm,
        "stddev_error_mm": stddev_error_mm,
        "macro_mean_error_mm": macro_mean,
        "macro_stddev_error_mm": macro_std,
        "mean_error_by_measurement": mean_error_by_measurement,
        "stddev_by_measurement": stddev_by_measurement,
        "count_by_measurement": count_by_measurement,
        "detailed_results": detailed_results,
    }
    with Path(output_path).open("w") as f:
        json.dump(results_out, f, indent=2)

    print(f"Evaluation results saved to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate distances between paired landmarks and compare measurements in mm.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/home/a332l/cluster-checkpoints/nnLandmark_baselines_results/SRLD/Dataset737_DMGLD_LFC/results"),
        help="Path to the directory containing predicted points JSON files."
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/all_landmarks_voxel.json"),
        help="Path to the ground truth JSON file."
    )
    parser.add_argument(
        "--spacing",
        type=Path,
        default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/spacing.json"),
        help="Path to the ground truth spacing JSON file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/a332l/cluster-checkpoints/nnLandmark_baselines_results/SRLD/Dataset737_DMGLD_LFC/results/measurement.json"),
        help="Path to save the evaluation results."
    )
    parser.add_argument("--max-landmark-index", type=int, default=6, help="Number of landmark measurement pairs to evaluate (i => landmark_i_1/landmark_i_2).")
    args = parser.parse_args()

    evaluate_measurements(args.results_dir, args.ground_truth, args.spacing, args.output, max_landmark_index=args.max_landmark_index)

if __name__ == "__main__":
    main()