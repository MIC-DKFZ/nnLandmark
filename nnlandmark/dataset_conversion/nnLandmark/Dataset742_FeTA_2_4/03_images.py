#!/usr/bin/env python3
import os
import argparse
import shutil
import json
from pathlib import Path
import SimpleITK as sitk


def load_transform(transform_path: Path) -> sitk.Transform:
    """
    Load an affine transformation from an Insight Transform File.

    Args:
        transform_path (Path): Path to the transform file.

    Returns:
        sitk.Transform: The loaded transformation.
    """
    try:
        # Attempt to load the transform using SimpleITK
        transform = sitk.ReadTransform(str(transform_path))
        return transform
    except Exception as e:
        print(f"[ERROR] Failed to load transform from {transform_path}: {e}")
        raise


def apply_affine_transform(image_path: Path, transform_path: Path, output_path: Path):
    """
    Apply affine transformation to an image using SimpleITK and save the result.

    Args:
        image_path (Path): Path to the input image.
        transform_path (Path): Path to the transform file.
        output_path (Path): Path to save the transformed image.
    """
    # Load the image
    image = sitk.ReadImage(str(image_path))

    # Load the transform
    transform = load_transform(transform_path)

    # Resample the image using the transform
    resampled_image = sitk.Resample(image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())

    # Save the transformed image
    sitk.WriteImage(resampled_image, str(output_path))
    print(f"Transformed image saved to: {output_path}")


def find_transform_file(subject_dir: Path, subject_id: str) -> Path:
    """
    Find the transform file for a subject, regardless of naming variations.

    Args:
        subject_dir (Path): Path to the subject's directory.
        subject_id (str): Subject ID (e.g., "sub-070").

    Returns:
        Path: Path to the transform file, or None if not found.
    """
    transform_candidates = list(subject_dir.glob(f"{subject_id}_rec-*_trf.txt"))
    if transform_candidates:
        return transform_candidates[0]  # Return the first matching transform file
    return None


def main():
    parser = argparse.ArgumentParser(description="Apply affine transformations to FeTA dataset images and copy to nnUNet raw imagesTr directory.")
    parser.add_argument(
        "--base",
        default="/path/to/2024_Ertl_nnLandmark/data/feta_2.4",
        help="Base FeTA dataset directory containing subject folders (e.g., sub-001/anat/...)"
    )
    parser.add_argument(
        "--target",
        default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/imagesTr/",
        help="Target directory to copy transformed images into"
    )
    parser.add_argument(
        "--landmark-json",
        default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/all_landmarks_voxel.json",
        help="Path to the landmark JSON file to filter cases with annotations."
    )
    args = parser.parse_args()

    base = Path(os.path.expanduser(args.base))
    target = Path(args.target)
    landmark_json_path = Path(args.landmark_json)

    # Load the landmark JSON to filter cases
    if not landmark_json_path.exists():
        raise FileNotFoundError(f"Landmark JSON file not found: {landmark_json_path}")
    with landmark_json_path.open("r") as f:
        landmark_data = json.load(f)

    # Get the list of cases with annotations
    cases_with_annotations = set(landmark_data.keys())

    target.mkdir(parents=True, exist_ok=True)

    for subject_dir in sorted(base.glob("derivatives/biometry/sub-*/anat")):
        subject_id = subject_dir.parent.name  # e.g., "sub-001"
        if subject_id not in cases_with_annotations:
            print(f"[INFO] Skipping {subject_id} as it has no annotations.")
            continue

        t2w_candidates = list(base.glob(f"{subject_id}/anat/*_T2w.nii*"))
        if not t2w_candidates:
            print(f"[WARN] No T2w image found for {subject_id}, skipping.")
            continue

        t2w_image = t2w_candidates[0]  # Take the first matching T2w image

        # Find the affine transformation file dynamically
        transform_path = find_transform_file(subject_dir, subject_id)
        if not transform_path:
            print(f"[WARN] No affine transformation found for {subject_id}, skipping.")
            continue

        # Define the output path
        output_path = target / f"{subject_id}_0000.nii.gz"

        # Apply the affine transformation
        try:
            apply_affine_transform(t2w_image, transform_path, output_path)
        except Exception as e:
            print(f"[ERROR] Failed to apply transform for {subject_id}: {e}")


if __name__ == "__main__":
    main()