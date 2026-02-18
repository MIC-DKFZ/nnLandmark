import json
from pathlib import Path

# Landmarks to keep
KEEP_LANDMARKS = {"landmark_3_1", "landmark_3_2", "landmark_5_1", "landmark_5_2", "landmark_6_1", "landmark_6_2"}

def filter_ground_truth(input_path: Path, output_path: Path) -> None:
    """Filter ground truth JSON to keep only specified landmarks."""
    with input_path.open("r") as f:
        ground_truth = json.load(f)

    filtered_gt = {}
    for case_id, landmarks in ground_truth.items():
        filtered_landmarks = {k: v for k, v in landmarks.items() if k in KEEP_LANDMARKS}
        if filtered_landmarks:
            filtered_gt[case_id] = filtered_landmarks

    with output_path.open("w") as f:
        json.dump(filtered_gt, f, indent=2)

    print(f"Filtered ground truth saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter ground truth JSON to keep only specific landmarks.")
    parser.add_argument(
        "--input",
        type=Path,
        default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_FeTA21/all_landmarks_voxel.json",
        help="Path to the input ground truth JSON file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_FeTA21/cerebellum_landmarks_voxel.json",
        help="Path to save the filtered ground truth JSON file."
    )
    args = parser.parse_args()

    filter_ground_truth(args.input, args.output)


if __name__ == "__main__":
    main()