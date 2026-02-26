#!/usr/bin/env python3
"""
Redo analysis of segmentation maps to ensure correct landmark counting.

- Analyze the original segmentation maps to verify that:
    - There are exactly 5 labels.
    - Each label has exactly 2 points (two single voxels).
- Adjust the landmarks to separate the two points for each label.
- Save the adjusted landmarks in JSON format.
- Create label maps in `labelsTr` with 3x3x3 cubes around each target voxel.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib


def analyze_segmentation(seg_path):
    """
    Analyze the segmentation map to extract voxel coordinates of landmarks.

    Args:
        seg_path (Path): Path to the segmentation map.

    Returns:
        dict: Dictionary of label -> list of voxel coordinates.
    """
    seg_img = nib.load(str(seg_path))
    seg_data = np.asarray(seg_img.dataobj)
    landmarks = defaultdict(list)

    for label in np.unique(seg_data):
        if label == 0:  # Skip background
            continue
        coords = np.argwhere(seg_data == label)
        if coords.shape[0] != 2:
            raise ValueError(f"Label {label} in {seg_path} does not have exactly 2 points (found {coords.shape[0]}).")
        for coord in coords:
            landmarks[int(label)].append(coord.tolist())

    return landmarks, seg_img.affine, seg_img.header


def separate_landmarks(landmarks):
    """
    Separate landmarks into individual points.

    Args:
        landmarks (dict): Original landmarks with voxel coordinates.

    Returns:
        dict: Adjusted landmarks with separated points.
    """
    adjusted_landmarks = {}
    for label, points in landmarks.items():
        if len(points) != 2:
            raise ValueError(f"Label {label} does not have exactly 2 points.")
        # Assign points as _1 and _2
        adjusted_landmarks[f"landmark{label}_1"] = points[0]
        adjusted_landmarks[f"landmark{label}_2"] = points[1]

    return adjusted_landmarks


def draw_cube(seg, center, half, label):
    """
    Draw a cube around the center voxel in the segmentation array.

    Args:
        seg (np.ndarray): Segmentation array.
        center (list): Center voxel coordinates.
        half (int): Half size of the cube.
        label (int): Label value to assign.
    """
    x, y, z = (int(round(c)) for c in center)
    seg[
        max(x - half, 0): x + half + 1,
        max(y - half, 0): y + half + 1,
        max(z - half, 0): z + half + 1,
    ] = label


def main():
    # Paths
    root = Path("/path/to/2024_Ertl_nnLandmark/data/feta_2.4")
    seg_dir = root / "derivatives/biometry"
    labels_tr_dir = Path("/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/labelsTr")
    labels_tr_dir.mkdir(parents=True, exist_ok=True)

    adjusted_landmarks = {}

    # Define the expected label mapping
    label_mapping = {
        "landmark1_1": 1, "landmark1_2": 2,
        "landmark2_1": 3, "landmark2_2": 4,
        "landmark3_1": 5, "landmark3_2": 6,
        "landmark4_1": 7, "landmark4_2": 8,
        "landmark5_1": 9, "landmark5_2": 10
    }

    for seg_path in sorted(seg_dir.glob("sub-*/anat/*_meas.nii.gz")):
        case_id = seg_path.parent.parent.name  # Extract case ID (e.g., "sub-001")
        print(f"Processing case: {case_id}")

        # Analyze segmentation to extract landmarks
        try:
            original_landmarks, affine, header = analyze_segmentation(seg_path)
        except ValueError as e:
            print(f"[ERROR] {e}")
            continue

        # Check if the case has exactly 5 landmarks
        if len(original_landmarks) != 5:
            print(f"[INFO] Skipping case {case_id} as it does not have exactly 5 landmarks (found {len(original_landmarks)}).")
            continue

        # Separate landmarks
        adjusted_case_landmarks = separate_landmarks(original_landmarks)
        adjusted_landmarks[case_id] = adjusted_case_landmarks

        # Create label map
        seg_shape = nib.load(str(seg_path)).shape
        seg = np.zeros(seg_shape, dtype=np.uint8)

        for landmark_name, coords in adjusted_case_landmarks.items():
            label = label_mapping[landmark_name]  # Use the predefined label mapping
            draw_cube(seg, coords, half=1, label=label)

        # Save label map
        label_map_path = labels_tr_dir / f"{case_id}.nii.gz"
        nib.save(nib.Nifti1Image(seg, affine, header), str(label_map_path))
        print(f"Saved label map for case {case_id} to {label_map_path}")

    # Save adjusted landmarks
    adjusted_landmarks_path = root / "adjusted_landmarks_voxel.json"
    with adjusted_landmarks_path.open("w") as f:
        json.dump(adjusted_landmarks, f, indent=2)
    print(f"Saved adjusted landmarks to {adjusted_landmarks_path}")


if __name__ == "__main__":
    main()