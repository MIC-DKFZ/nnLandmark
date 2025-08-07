import os
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

import nibabel as nib

def get_nifti_orientation(nifti_file):
    img = nib.load(str(nifti_file))
    ornt = nib.orientations.io_orientation(img.affine)
    code = nib.orientations.ornt2axcodes(ornt)
    return "".join(code)

def parse_fcsv(fcsv_path):
    coords = []
    with open(fcsv_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split(',')
            try:
                x, y, z = float(fields[1]), float(fields[2]), float(fields[3])
                coords.append([x, y, z])
            except Exception as e:
                print(f"[WARN] Could not parse row in {fcsv_path}: {line.strip()}")
    return np.array(coords)

def is_missing(row):
    return (
        np.isnan(row).any()
    )

def scan_afids_dataset(root_path, spacing_tol=0.01):
    datasets_path = Path(root_path)
    for dataset_folder in sorted(datasets_path.iterdir()):
        if not dataset_folder.is_dir():
            continue
        print(f"\n[DATASET] {dataset_folder.name}")
        per_case_landmark_counts = Counter()
        case_missing_details = defaultdict(list)
        niftis_spacing_counter = Counter()
        niftis_spacing_examples = {}
        num_cases = 0
        orientations_counter = Counter()
        orientations_examples = {}


        # Find all annotation .fcsv files
        fcsv_files = list(dataset_folder.glob("derivatives/afids_groundtruth/sub-*/anat/*.fcsv"))
        for fcsv in sorted(fcsv_files):
            num_cases += 1

            # --- Landmark analysis ---
            coords = parse_fcsv(fcsv)
            n_landmarks = 0
            for idx, row in enumerate(coords):
                if is_missing(row):
                    case_missing_details[fcsv.name].append((idx, row))
                else:
                    n_landmarks += 1
            per_case_landmark_counts[n_landmarks] += 1

            # --- NIfTI spacing analysis ---
            # Extract subject ID from path: .../sub-XXXXX/anat/*.fcsv
            try:
                # Get subject dir from annotation path
                # .../derivatives/afids_groundtruth/sub-103111/anat/....fcsv
                subject_dir = fcsv.parents[1].name  # should be 'sub-103111'
                nifti_folder = dataset_folder / subject_dir / "anat"
                nifti_files = sorted(nifti_folder.glob("*.nii*"))
                if not nifti_files:
                    print(f"[WARN] No NIfTI found for {fcsv.name} in {nifti_folder}")
                    continue
                nifti_file = nifti_files[0]
                img = nib.load(str(nifti_file))

                #spacing
                spacing = tuple(np.round(img.header.get_zooms(), 5))
                rounded_spacing = tuple(round(v / spacing_tol) * spacing_tol for v in spacing)
                niftis_spacing_counter[rounded_spacing] += 1
                if rounded_spacing not in niftis_spacing_examples:
                    niftis_spacing_examples[rounded_spacing] = nifti_file.name

                # orientation
                orientation = get_nifti_orientation(nifti_file)
                orientations_counter[orientation] += 1
                if orientation not in orientations_examples:
                    orientations_examples[orientation] = nifti_file.name

            except Exception as e:
                print(f"[WARN] Error reading NIfTI for {fcsv.name}: {e}")

        # Print results for this dataset
        print(f"Total cases: {num_cases}")
        print("Histogram of annotated landmarks per case:")
        for n, count in sorted(per_case_landmark_counts.items()):
            print(f"  {n} landmarks: {count} cases")
        if case_missing_details:
            print("Cases with missing/invalid landmarks:")
            for fname, issues in case_missing_details.items():
                print(f"  {fname}: {len(issues)} missing/invalid landmarks")
        else:
            print("All cases have only valid landmark coordinates!")

        print("\nHistogram of NIfTI voxel spacings (rounded):")
        for sp, count in sorted(niftis_spacing_counter.items()):
            print(f"  {sp} -> {count} cases")
        print("Examples per spacing:")
        for sp, fname in niftis_spacing_examples.items():
            print(f"  {sp}: {fname}")

        print("\nNIfTI orientation codes (e.g. RAS, LPI):")
        for code, count in orientations_counter.items():
            print(f"  {code}: {count} cases (e.g. {orientations_examples[code]})")


if __name__ == "__main__":
    root = "/home/a332l/dev/Project_SoftDiceLoss/data/afids-data/data/datasets/"
    scan_afids_dataset(root, spacing_tol=0.01)


