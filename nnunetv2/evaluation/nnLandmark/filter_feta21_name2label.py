import json
from pathlib import Path

# Define the new mapping for the landmarks
NEW_MAPPING = {
    "landmark_3_1": 1,
    "landmark_3_2": 2,
    "landmark_5_1": 3,
    "landmark_5_2": 4,
    "landmark_6_1": 5,
    "landmark_6_2": 6
}

def update_name_to_label(input_path: Path, output_path: Path, new_mapping: dict) -> None:
    """Update the name_to_label.json file with the new mapping."""
    with input_path.open("r") as f:
        original_mapping = json.load(f)

    # Filter and remap the original mapping
    updated_mapping = {name: new_mapping[name] for name in new_mapping if name in original_mapping}

    # Save the updated mapping
    with output_path.open("w") as f:
        json.dump(updated_mapping, f, indent=2)

    print(f"Updated name_to_label.json saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Update name_to_label.json to remap labels.")
    parser.add_argument(
        "--input",
        type=Path,
        default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_FeTA21/name_to_label.json",
        help="Path to the original name_to_label.json file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_FeTA21/cerebellum_name_to_label.json",
        help="Path to save the updated name_to_label.json file."
    )
    args = parser.parse_args()

    update_name_to_label(args.input, args.output, NEW_MAPPING)


if __name__ == "__main__":
    main()