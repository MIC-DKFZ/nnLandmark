#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Sequence, Any, Optional
from collections import defaultdict


def load_json(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _to_mm_distance(vec_vox: Sequence[float], spacing: Sequence[float]) -> float:
    return float(np.linalg.norm(np.array(vec_vox, dtype=float) * np.array(spacing, dtype=float)))


def _recalc_filtered_summary(orig_summary: Dict[str, Any], keep_set: set) -> Dict[str, Any]:
    """
    Filter original summary's detailed_results to keep only classes in keep_set,
    then recompute MRE_by_class, std_by_class, MRE_micro, std_micro, std_macro_class,
    and SRD_xmm (x=2,3,4) based on the filtered detailed_results.
    """
    out = dict(orig_summary)  # shallow copy of top-level keys; will replace computed fields

    orig_detailed = orig_summary.get("detailed_results", {})
    filtered_detailed: Dict[str, Dict[str, float]] = {}

    # collect per-class errors
    per_class_errors: Dict[str, List[float]] = defaultdict(list)
    all_errors: List[float] = []

    for case_id, case_map in orig_detailed.items():
        if not isinstance(case_map, dict):
            continue
        filtered_case = {}
        for cls, val in case_map.items():
            if cls in keep_set:
                try:
                    v = float(val)
                except Exception:
                    continue
                filtered_case[cls] = v
                per_class_errors[cls].append(v)
                all_errors.append(v)
        if filtered_case:
            filtered_detailed[case_id] = filtered_case

    # Per-class stats
    MRE_by_class: Dict[str, float] = {}
    std_by_class: Dict[str, float] = {}
    class_means: List[float] = []

    for cls, errs in per_class_errors.items():
        arr = np.array(errs, dtype=float)
        mean = float(arr.mean()) if arr.size > 0 else 0.0
        sd = float(arr.std()) if arr.size > 0 else 0.0
        MRE_by_class[cls] = mean
        std_by_class[cls] = sd
        class_means.append(mean)

    # Micro / macro stats
    all_arr = np.array(all_errors, dtype=float) if all_errors else np.array([], dtype=float)
    MRE_micro = float(all_arr.mean()) if all_arr.size > 0 else 0.0
    std_micro = float(all_arr.std()) if all_arr.size > 0 else 0.0
    std_macro_class = float(np.array(class_means, dtype=float).std()) if len(class_means) > 0 else 0.0

    # SRD thresholds (success rate at thresholds)
    srd = {}
    for thr in (2.0, 3.0, 4.0):
        if all_arr.size:
            srd[f"SRD_{int(thr)}mm"] = float((all_arr <= thr).sum() / all_arr.size)
        else:
            srd[f"SRD_{int(thr)}mm"] = 0.0

    # build output summary: keep original other keys but replace computed fields
    out["detailed_results"] = filtered_detailed
    out["MRE_by_class"] = MRE_by_class
    out["std_by_class"] = std_by_class
    out["MRE_micro"] = MRE_micro
    out["std_micro"] = std_micro
    out["std_macro_class"] = std_macro_class
    # overwrite SRD_* keys if present, else add
    for k, v in srd.items():
        # original used keys "SRD_2mm" etc. Keep same naming style
        keyname = k.replace("SRD_", "SRD_")
        out[keyname] = v

    return out


def evaluate_measurements(
    predictions_path: Path,
    ground_truth_path: Path,
    spacing_path: Path,
    output_path: Path,
    summary_path: Optional[Path] = None,
):
    # keep only these landmark index groups (1/2 pairs per index)
    keep_indices: List[int] = [3, 5, 6]
    keep_set = {f"landmark_{i}_1" for i in keep_indices} | {f"landmark_{i}_2" for i in keep_indices}

    predictions = load_json(predictions_path)
    ground_truth = load_json(ground_truth_path)
    spacing = load_json(spacing_path)

    # filter predictions and write filtered JSON with suffix "_cerebellum"
    filtered_predictions = {}
    for case_id, lm_map in predictions.items():
        filtered = {k: v for k, v in lm_map.items() if k in keep_set}
        if filtered:
            filtered_predictions[case_id] = filtered
    pred_filtered_path = predictions_path.with_name(predictions_path.stem + "_cerebellum" + predictions_path.suffix)
    write_json(filtered_predictions, pred_filtered_path)
    print(f"Wrote filtered predictions: {pred_filtered_path}")

    # handle summary_mm: find provided or sibling summary_mm.json
    summary_in: Optional[Path] = None
    if summary_path is not None:
        summary_in = summary_path
    else:
        sibling = predictions_path.parent / "summary_mm.json"
        if sibling.exists():
            summary_in = sibling

    if summary_in is not None and summary_in.exists():
        orig_summary = load_json(summary_in)
        filtered_summary = _recalc_filtered_summary(orig_summary, keep_set)
        summary_out_path = summary_in.with_name(summary_in.stem + "_cerebellum" + summary_in.suffix)
        write_json(filtered_summary, summary_out_path)
        print(f"Wrote filtered & recalculated summary: {summary_out_path}")
    else:
        print("No summary_mm.json found or provided; skipping summary filtering/recalc.")

    # now compute measurement errors for the selected pairs (3/5/6) and save per-case results
    measurement_errors: Dict[str, List[float]] = {}
    detailed_results: Dict[str, Dict[str, Any]] = {}

    for case_id, gt_case in ground_truth.items():
        if case_id not in spacing:
            print(f"[WARN] spacing missing for {case_id} — skipping")
            continue
        sx, sy, sz = spacing[case_id].get("image_spacing", [None, None, None])
        if None in (sx, sy, sz):
            print(f"[WARN] incomplete spacing for {case_id} — skipping")
            continue
        pred_case = filtered_predictions.get(case_id, {})
        case_res = {"ground_truth_measurements_mm": {}, "predicted_measurements_mm": {}, "errors_mm": {}}

        for i in keep_indices:
            lm1 = f"landmark_{i}_1"
            lm2 = f"landmark_{i}_2"
            key = f"{lm1}-{lm2}"

            if lm1 in gt_case and lm2 in gt_case:
                gt_vec = np.array(gt_case[lm1]) - np.array(gt_case[lm2])
                gt_mm = _to_mm_distance(gt_vec, (sx, sy, sz))
                case_res["ground_truth_measurements_mm"][key] = float(gt_mm)

                if lm1 in pred_case and lm2 in pred_case:
                    pred_vec = np.array(pred_case[lm1]) - np.array(pred_case[lm2])
                    pred_mm = _to_mm_distance(pred_vec, (sx, sy, sz))
                    case_res["predicted_measurements_mm"][key] = float(pred_mm)

                    err = abs(gt_mm - pred_mm)
                    case_res["errors_mm"][key] = float(err)
                    measurement_errors.setdefault(key, []).append(float(err))
        detailed_results[case_id] = case_res

    all_errors = [e for errs in measurement_errors.values() for e in errs]
    mean_error = float(np.mean(all_errors)) if all_errors else 0.0
    std_error = float(np.std(all_errors)) if all_errors else 0.0

    stats = {
        "mean_error_mm": mean_error,
        "stddev_error_mm": std_error,
        "per_measurement": {},
        "detailed_results": detailed_results,
    }
    for k, errs in measurement_errors.items():
        arr = np.array(errs, dtype=float)
        stats["per_measurement"][k] = {
            "mean_mm": float(arr.mean()),
            "std_mm": float(arr.std()),
            "count": int(arr.size),
        }

    out_path = output_path.with_name(output_path.stem + "_cerebellum" + output_path.suffix)
    write_json(stats, out_path)
    print(f"Wrote measurement results: {out_path}")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Evaluate selected cerebellum landmark measurements (3,5,6 pairs).")
    p.add_argument("--predictions", type=Path, default="/home/a332l/cluster-checkpoints/nnLandmark_baselines_results/landmarker/Dataset738_FeTA21/predictions/prediction_all_landmark_voxel.json", help="Predictions JSON (voxel coords)")
    p.add_argument("--ground_truth", type=Path, default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_FeTA21/all_landmarks_voxel.json", help="Ground truth JSON (voxel coords)")
    p.add_argument("--spacing", type=Path, default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_FeTA21/spacing.json", help="Spacing JSON")
    p.add_argument("--output", type=Path, default="/home/a332l/cluster-checkpoints/nnLandmark_baselines_results/landmarker/Dataset738_FeTA21/predictions/measurement_results_cerebellum.json", help="Output measurements JSON")
    p.add_argument("--summary", type=Path, default="/home/a332l/cluster-checkpoints/nnLandmark_baselines_results/landmarker/Dataset738_FeTA21/predictions/summary_mm.json", help="Optional: summary_mm.json to filter and save with _cerebellum suffix")
    args = p.parse_args()

    evaluate_measurements(args.predictions, args.ground_truth, args.spacing, args.output, args.summary)

if __name__ == "__main__":
    main()