#!/usr/bin/env python3
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


def load_json(file_path: Path) -> Dict[str, Any]:
    with file_path.open() as f:
        return json.load(f)


def calculate_distance_mm(
    coord1_vox: Tuple[float, float, float],
    coord2_vox: Tuple[float, float, float],
    spacing_xyz: Tuple[float, float, float],
) -> float:
    """
    Physically correct distance in mm for anisotropic spacing:
      dist_mm = || (coord1 - coord2) * spacing ||_2
    """
    c1 = np.asarray(coord1_vox, dtype=float)
    c2 = np.asarray(coord2_vox, dtype=float)
    sp = np.asarray(spacing_xyz, dtype=float)
    diff_mm = (c1 - c2) * sp
    return float(np.sqrt(np.sum(diff_mm ** 2)))


def extract_spacing_from_entry(entry) -> Tuple[float, float, float]:
    """
    Accepts:
      - (sx,sy,sz) list/tuple
      - dict with keys: image_spacing / annotation_spacing / spacing
    """
    if entry is None:
        raise KeyError("Spacing entry is None")

    if isinstance(entry, (list, tuple)) and len(entry) == 3:
        return float(entry[0]), float(entry[1]), float(entry[2])

    if isinstance(entry, dict):
        for k in ("image_spacing", "annotation_spacing", "spacing"):
            if k in entry and entry[k] is not None:
                vals = entry[k]
                return float(vals[0]), float(vals[1]), float(vals[2])

    raise ValueError(f"Unrecognized spacing entry: {entry}")


def load_pred_spacing_from_pkl(spacing_pkl_dir: Path, case_id: str) -> Tuple[float, float, float]:
    """
    Loads per-case spacing for predictions from:
      spacing_pkl_dir/{case_id}.pkl  with key 'rawspacing'
    """
    p = spacing_pkl_dir / f"{case_id}.pkl"
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and obj.get("rawspacing") is not None:
        vals = obj["rawspacing"]
        if isinstance(vals, (list, tuple)) and len(vals) == 3:
            return float(vals[0]), float(vals[1]), float(vals[2])

    raise ValueError(f"No rawspacing in pkl for case {case_id}: {p}")


def build_predicted_points_from_file(
    predicted_data: Dict[str, Any]
) -> Tuple[Dict[str, Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """
    Supports common JSON layouts:
      1) {"predicted_points": {...}, "spacing": ...}
      2) {"spacing": ..., "landmark_1_1": ..., ...} or landmark_* as dict with 'coordinates'
      3) {"landmark_1_1": [x,y,z], ...} (no spacing)
    Returns:
      (pred_pts, pred_spacing_in_json_or_None)
    """
    if "predicted_points" in predicted_data:
        pred_pts = predicted_data["predicted_points"]
        pred_spacing = None
        if "spacing" in predicted_data:
            pred_spacing = extract_spacing_from_entry(predicted_data["spacing"])
        return pred_pts, pred_spacing

    if "spacing" in predicted_data and any(k.startswith("landmark_") for k in predicted_data.keys()):
        pred_spacing = extract_spacing_from_entry(predicted_data.get("spacing"))
        pred_pts = {}
        for k, v in predicted_data.items():
            if not k.startswith("landmark_"):
                continue
            if isinstance(v, dict) and "coordinates" in v:
                pred_pts[k] = tuple(v["coordinates"])
            elif isinstance(v, (list, tuple)) and len(v) == 3:
                pred_pts[k] = tuple(v)
        return pred_pts, pred_spacing

    pred_pts = {}
    for k, v in predicted_data.items():
        if k.startswith("landmark_") and isinstance(v, (list, tuple)) and len(v) == 3:
            pred_pts[k] = tuple(v)
    if pred_pts:
        return pred_pts, None

    raise ValueError("Unrecognized predicted JSON structure")


def evaluate_case(
    case_id: str,
    predicted_points: Dict[str, Tuple[float, float, float]],
    pred_spacing: Tuple[float, float, float],
    ground_truth_points: Dict[str, Tuple[float, float, float]],
    gt_spacing: Tuple[float, float, float],
    max_landmark_index: int,
) -> Dict[str, Any]:
    case_results = {
        "ground_truth_measurements_mm": {},
        "predicted_measurements_mm": {},
        "errors_mm": {},
        "gt_spacing": list(map(float, gt_spacing)),
        "pred_spacing": list(map(float, pred_spacing)),
    }

    for i in range(1, max_landmark_index + 1):
        name1 = f"landmark_{i}_1"
        name2 = f"landmark_{i}_2"
        key = f"{name1}-{name2}"

        if name1 not in ground_truth_points or name2 not in ground_truth_points:
            continue

        gt_mm = calculate_distance_mm(ground_truth_points[name1], ground_truth_points[name2], gt_spacing)
        case_results["ground_truth_measurements_mm"][key] = float(np.round(gt_mm, 6))

        if name1 in predicted_points and name2 in predicted_points:
            pred_mm = calculate_distance_mm(predicted_points[name1], predicted_points[name2], pred_spacing)
            case_results["predicted_measurements_mm"][key] = float(np.round(pred_mm, 6))
            case_results["errors_mm"][key] = float(np.round(abs(gt_mm - pred_mm), 6))

    return case_results


def evaluate_measurements(
    results_dir: Path,
    ground_truth_path: Path,
    gt_spacing_json_path: Path,
    pred_spacing_pkl_dir: Path,
    output_path: Path,
    max_landmark_index: int = 6,
):
    """
    - Predictions: voxel coords; spacing from per-case PKL (rawspacing), unless the prediction JSON already contains spacing.
    - Ground truth: voxel coords; spacing from gt_spacing_json_path (spacing.json -> image_spacing).
    Produces:
      - per-case measurements and errors
      - micro mean/std across all errors
      - macro mean/std across measurement classes (mean over per-class means)
    """
    results_dir = Path(results_dir)
    ground_truth = load_json(ground_truth_path)
    gt_spacing_json = load_json(gt_spacing_json_path)

    if pred_spacing_pkl_dir is None or not pred_spacing_pkl_dir.exists():
        raise SystemExit(f"pred_spacing_pkl_dir not found: {pred_spacing_pkl_dir}")

    detailed_results: Dict[str, Any] = {}

    def get_gt_spacing(case_id: str) -> Tuple[float, float, float]:
        if case_id not in gt_spacing_json:
            raise KeyError(f"Case {case_id} not in GT spacing JSON")
        return extract_spacing_from_entry(gt_spacing_json[case_id]["image_spacing"])

    def get_pred_spacing_from_pkl(case_id: str) -> Tuple[float, float, float]:
        return load_pred_spacing_from_pkl(pred_spacing_pkl_dir, case_id)

    # --- aggregated predictions file (optional) ---
    agg_file = results_dir / "prediction_all_landmark_voxel.json"
    if agg_file.exists():
        agg_preds = load_json(agg_file)
        for case_id, pred_map in agg_preds.items():
            if case_id not in ground_truth:
                print(f"[WARN] case {case_id} in aggregated predictions not found in GT, skipping")
                continue
            try:
                gt_spacing = get_gt_spacing(case_id)
            except Exception as e:
                print(f"[WARN] could not load GT spacing for {case_id}: {e}; skipping")
                continue
            try:
                pred_spacing = get_pred_spacing_from_pkl(case_id)
            except Exception as e:
                print(f"[WARN] could not load pred spacing pkl for {case_id}: {e}; skipping")
                continue

            case_res = evaluate_case(
                case_id=case_id,
                predicted_points=pred_map,
                pred_spacing=pred_spacing,
                ground_truth_points=ground_truth[case_id],
                gt_spacing=gt_spacing,
                max_landmark_index=max_landmark_index,
            )
            detailed_results[case_id] = case_res

    # --- per-case prediction jsons ---
    for predicted_file in sorted(results_dir.glob("*.json")):
        if predicted_file.name in ("prediction_all_landmark_voxel.json", output_path.name):
            continue

        try:
            pd = load_json(predicted_file)
        except Exception as e:
            print(f"[WARN] failed to load {predicted_file}: {e}")
            continue

        case_id = predicted_file.stem.split("_")[0]
        if case_id not in ground_truth:
            print(f"[WARN] Case '{case_id}' not found in ground truth. Skipping.")
            continue

        try:
            gt_spacing = get_gt_spacing(case_id)
        except Exception as e:
            print(f"[WARN] GT spacing for case '{case_id}' not found or invalid: {e}. Skipping.")
            continue

        try:
            pred_pts, pred_spacing_in_json = build_predicted_points_from_file(pd)
        except ValueError:
            print(f"[WARN] Unrecognized predicted file structure for {predicted_file}, skipping")
            continue

        # prefer spacing in prediction file; else use PKL rawspacing
        try:
            pred_spacing = pred_spacing_in_json or get_pred_spacing_from_pkl(case_id)
        except Exception as e:
            print(f"[WARN] Pred spacing for case '{case_id}' not found/invalid: {e}. Skipping.")
            continue

        case_res = evaluate_case(
            case_id=case_id,
            predicted_points=pred_pts,
            pred_spacing=pred_spacing,
            ground_truth_points=ground_truth[case_id],
            gt_spacing=gt_spacing,
            max_landmark_index=max_landmark_index,
        )
        detailed_results[case_id] = case_res

    # --- aggregate statistics ---
    errors_by_measurement: Dict[str, list] = {}
    all_errors: list = []

    for case_res in detailed_results.values():
        for meas_name, err in case_res.get("errors_mm", {}).items():
            errors_by_measurement.setdefault(meas_name, []).append(float(err))
            all_errors.append(float(err))

    mean_error_mm = float(np.round(np.mean(all_errors), 6)) if all_errors else 0.0
    stddev_error_mm = float(np.round(np.std(all_errors), 6)) if all_errors else 0.0

    mean_error_by_measurement: Dict[str, float] = {}
    stddev_by_measurement: Dict[str, float] = {}
    count_by_measurement: Dict[str, int] = {}

    for meas_name, errs in sorted(errors_by_measurement.items()):
        errs_arr = np.asarray(errs, dtype=float)
        mean_error_by_measurement[meas_name] = float(np.round(np.mean(errs_arr), 6))
        stddev_by_measurement[meas_name] = float(np.round(np.std(errs_arr), 6))
        count_by_measurement[meas_name] = int(errs_arr.size)

    class_means = np.asarray(list(mean_error_by_measurement.values()), dtype=float)
    macro_mean = float(np.round(class_means.mean(), 6)) if class_means.size else None
    macro_std = float(np.round(class_means.std(ddof=0), 6)) if class_means.size else None

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results_out, f, indent=2)

    print(f"Evaluation results saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate paired landmark distances (mm) using GT spacing from spacing.json and prediction spacing from per-case PKL rawspacing."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/baselines/SRLD/Dataset737_DMGLD_LFC/results"),
        help="Directory containing prediction JSON files (and optionally prediction_all_landmark_voxel.json).",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/all_landmarks_voxel.json"),
        help="Ground truth landmarks JSON (voxel coordinates).",
    )
    parser.add_argument(
        "--gt-spacing-json",
        type=Path,
        default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC/spacing.json"),
        help="spacing.json with original image spacing per case (uses key image_spacing).",
    )
    parser.add_argument(
        "--pred-spacing-pkl-dir",
        type=Path,
        default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/baslines_data/SRLD_format/Dataset737_DMGLD_LFC/labelsTs"),
        help="Directory containing per-case {case_id}.pkl with key 'rawspacing' (prediction spacing).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/baselines/SRLD/Dataset737_DMGLD_LFC/results/measurement.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-landmark-index",
        type=int,
        default=6,
        help="Evaluate pairs landmark_i_1/landmark_i_2 for i=1..max-landmark-index.",
    )
    args = parser.parse_args()

    evaluate_measurements(
        results_dir=args.results_dir,
        ground_truth_path=args.ground_truth,
        gt_spacing_json_path=args.gt_spacing_json,
        pred_spacing_pkl_dir=args.pred_spacing_pkl_dir,
        output_path=args.output,
        max_landmark_index=args.max_landmark_index,
    )


if __name__ == "__main__":
    main()
