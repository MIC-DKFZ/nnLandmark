#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_json(file_path: Path) -> Dict:
    """Load a JSON file."""
    with file_path.open() as f:
        return json.load(f)


def calculate_euclidean_distance(coord1: Tuple[float, float, float], coord2: Tuple[float, float, float]) -> float:
    """Calculate the Euclidean distance between two 3D coordinates."""
    return float(np.sqrt(np.sum((np.array(coord1) - np.array(coord2)) ** 2)))


def evaluate_measurements(predictions_path: Path, ground_truth_path: Path, spacing_path: Path, output_path: Path):
    """
    Evaluate distances between paired landmarks and compare measurements in mm.
    Args:
        predictions_path: Path to the predictions JSON (in voxel space).
        ground_truth_path: Path to the ground truth JSON (in voxel space) in nnUNet_raw.
        spacing_path: Path to the spacing JSON in nnUNet_raw.
        output_path: Path to save the evaluation results.
    """
    # Load predictions, ground truth, and spacing
    predictions = load_json(predictions_path)
    ground_truth = load_json(ground_truth_path)
    spacing = load_json(spacing_path)

    # Initialize results
    measurement_errors = {}  # {measurement_key: [errors_mm]}
    detailed_results = {}

    # Iterate over cases
    for case_id in ground_truth.keys():
        if case_id not in predictions:
            print(f"[WARN] Case '{case_id}' not found in predictions. Skipping.")
            continue
        if case_id not in spacing:
            print(f"[WARN] Spacing for case '{case_id}' not found. Skipping.")
            continue

        gt_case = ground_truth[case_id]
        pred_case = predictions[case_id]
        case_spacing = spacing[case_id]["image_spacing"]
        sx, sy, sz = case_spacing  # Spacing in mm for x, y, z
        case_results = {"ground_truth_measurements_mm": {}, "predicted_measurements_mm": {}, "errors_mm": {}}

        # Iterate over paired landmarks
        for i in range(1, 100):  # Assuming landmark indices go up to 99
            landmark_1 = f"landmark{i}_1"
            landmark_2 = f"landmark{i}_2"
            measurement_key = f"{landmark_1}-{landmark_2}"

            if landmark_1 in gt_case and landmark_2 in gt_case:
                # Ground truth measurement in voxel space
                gt_distance_voxel = calculate_euclidean_distance(gt_case[landmark_1], gt_case[landmark_2])
                # Convert ground truth measurement to mm (isotropic approximation)
                gt_distance_mm = gt_distance_voxel * np.sqrt(sx**2 + sy**2 + sz**2)
                case_results["ground_truth_measurements_mm"][measurement_key] = gt_distance_mm

                # Predicted measurement in voxel space
                if landmark_1 in pred_case and landmark_2 in pred_case:
                    pred_distance_voxel = calculate_euclidean_distance(pred_case[landmark_1], pred_case[landmark_2])
                    # Convert predicted measurement to mm
                    pred_distance_mm = pred_distance_voxel * np.sqrt(sx**2 + sy**2 + sz**2)
                    case_results["predicted_measurements_mm"][measurement_key] = pred_distance_mm

                    # Error in measurement
                    error_mm = abs(gt_distance_mm - pred_distance_mm)
                    case_results["errors_mm"][measurement_key] = error_mm
                    
                    # Collect errors per measurement class for aggregation
                    measurement_errors.setdefault(measurement_key, []).append(error_mm)

        detailed_results[case_id] = case_results

    # Aggregate all errors (overall)
    all_errors_mm = [
        error
        for case in detailed_results.values()
        for error in case["errors_mm"].values()
    ]
    mean_error_mm = np.mean(all_errors_mm) if all_errors_mm else 0.0
    stddev_error_mm = np.std(all_errors_mm) if all_errors_mm else 0.0

    # Compute per-measurement statistics (mean, std, count)
    mean_error_by_measurement = {}
    stddev_error_by_measurement = {}
    count_by_measurement = {}
    for measurement_key, errors in measurement_errors.items():
        errors_array = np.array(errors)
        mean_error_by_measurement[measurement_key] = float(np.mean(errors_array))
        stddev_error_by_measurement[measurement_key] = float(np.std(errors_array))
        count_by_measurement[measurement_key] = int(len(errors_array))

    # Save results
    results = {
        "mean_error_mm": float(mean_error_mm),
        "stddev_error_mm": float(stddev_error_mm),
        "mean_error_by_measurement": mean_error_by_measurement,
        "stddev_error_by_measurement": stddev_error_by_measurement,
        "count_by_measurement": count_by_measurement,
        "detailed_results": detailed_results,
    }
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to {output_path}")
    print("\nPer-measurement statistics:")
    for meas, mean_err in mean_error_by_measurement.items():
        std_err = stddev_error_by_measurement[meas]
        count = count_by_measurement[meas]
        print(f"  {meas}: {mean_err:.2f} ± {std_err:.2f} mm (n={count})")



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate distances between paired landmarks and compare measurements in mm.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="/path/to/evaluation/Dataset742_FeTA_2_4/nnLandmark_trainer/prediction/renamed_landmarks.json"
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="/path/to/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/all_landmarks_voxel.json"
    )
    parser.add_argument(
        "--spacing",
        type=Path,
        required=True,
        help="/path/to/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/spacing.json."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="/path/to/evaluation/Dataset742_FeTA_2_4/nnLandmark_trainer/prediction/measurement.json"
    )
    args = parser.parse_args()

    evaluate_measurements(args.predictions, args.ground_truth, args.spacing, args.output)


if __name__ == "__main__":
    main()