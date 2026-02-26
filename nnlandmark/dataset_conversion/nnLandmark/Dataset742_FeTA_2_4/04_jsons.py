#!/usr/bin/env python3
import json
from pathlib import Path
import nibabel as nib


def main():
    # Paths
    dataset_root = Path("/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4")
    images_dir = dataset_root / "imagesTr"
    landmarks_json_path = dataset_root / "all_landmarks_voxel.json"

    # Load the landmarks JSON
    if not landmarks_json_path.exists():
        raise FileNotFoundError(f"Landmarks JSON file not found: {landmarks_json_path}")
    with landmarks_json_path.open("r") as f:
        all_landmarks = json.load(f)

    # Initialize outputs
    out_spacing = {}
    name_to_label = {}

    # Process each case
    for case_id, landmarks in all_landmarks.items():
        image_path = images_dir / f"{case_id}_0000.nii.gz"
        if not image_path.exists():
            print(f"[WARN] Image file not found for case {case_id}, skipping.")
            continue

        # Read image spacing from NIfTI header
        img = nib.load(str(image_path))
        spacing = img.header.get_zooms()[:3]  # Extract voxel spacing (x, y, z)

        # Add spacing information
        out_spacing[case_id] = {
            "image_spacing": [float(s) for s in spacing],
            "annotation_spacing": [float(s) for s in spacing],  # Assuming annotation spacing matches image spacing
        }

        # Build name-to-label mapping
        for name in landmarks:
            if name not in name_to_label:
                tail = name.split("_")[-1]
                label = int(tail) if tail.isdigit() else len(name_to_label) + 1
                name_to_label[name] = label

    # Write outputs
    (dataset_root / "spacing.json").write_text(
        json.dumps(out_spacing, indent=2)
    )
    (dataset_root / "name_to_label.json").write_text(
        json.dumps(name_to_label, indent=2)
    )

    print(f"✅  Wrote spacing and name_to_label JSONs to {dataset_root}")


if __name__ == "__main__":
    main()