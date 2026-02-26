import os
from pathlib import Path
import numpy as np
import nibabel as nib
import json
from collections import Counter, defaultdict

def is_missing(row):
    return (
        any(np.isnan(x) for x in row) or
        all(np.isclose(x, 0) for x in row) or
        all(np.isclose(x, -1) for x in row) or
        any(x < 0 for x in row)
    )

def get_nifti_orientation(nifti_file):
    img = nib.load(str(nifti_file))
    ornt = nib.orientations.io_orientation(img.affine)
    code = nib.orientations.ornt2axcodes(ornt)
    return "".join(code)

def main():
    # ---- Paths ----
    base_dir = "/home/a332l/dev/Project_nnLandmark/data/landmark_datasets/face_dataset/"
    vol_dir = Path(base_dir) / "volume"
    anno_dir = Path(base_dir) / "landmark_anno"

    # ---- Grab both .nii and .nii.gz ----
    volumes = sorted(
        list(vol_dir.glob("*.nii")) +
        list(vol_dir.glob("*.nii.gz"))
    )
    labels = sorted(anno_dir.glob("*_landmark.json"))

    # ---- Indexing ----
    vol_stems = {}
    for v in volumes:
        name = v.name
        if name.endswith('.nii.gz'):
            case_id = name[:-7]
        elif name.endswith('.nii'):
            case_id = name[:-4]
        else:
            continue
        vol_stems[case_id] = v

    label_stems = {l.stem.replace("_landmark", ""): l for l in labels}
    all_cases = set(vol_stems) | set(label_stems)

    # ---- Counters & Collectors ----
    spacing_counter_nifti    = Counter()
    spacing_examples_nifti   = {}
    spacing_counter_json     = Counter()
    spacing_examples_json    = {}
    unit_counter_json        = Counter()
    unit_examples_json       = {}

    orientation_counter      = Counter()
    orientation_examples     = {}

    landmark_count_hist      = Counter()
    cases_missing_landmarks  = defaultdict(list)
    images_without_label     = []
    labels_without_image     = []
    spacing_mismatches       = []  # list of (case, nifti_spacing, json_spacing)

    for case in sorted(all_cases):
        vol  = vol_stems.get(case)
        anno = label_stems.get(case)

        if not vol:
            labels_without_image.append(case)
            continue
        if not anno:
            images_without_label.append(case)
            continue

        # --- NIfTI info ---
        try:
            img = nib.load(str(vol))
            spacing_nifti = tuple(np.round(img.header.get_zooms(), 5))
            spacing_counter_nifti[spacing_nifti] += 1
            spacing_examples_nifti.setdefault(spacing_nifti, vol.name)

            orient = get_nifti_orientation(vol)
            orientation_counter[orient] += 1
            orientation_examples.setdefault(orient, vol.name)
        except Exception as e:
            print(f"[WARN] Could not read NIfTI {vol.name}: {e}")
            continue

        # --- Landmark JSON info & unit check ---
        try:
            with open(anno, 'r') as f:
                data = json.load(f)

            # spacing value
            spacing_json_val = data.get("spacing", None)
            if spacing_json_val is not None:
                spacing_json = (spacing_json_val,)*3
                spacing_counter_json[spacing_json] += 1
                spacing_examples_json.setdefault(spacing_json, anno.name)
                # detect mismatch
                if spacing_nifti != spacing_json:
                    spacing_mismatches.append((case, spacing_nifti, spacing_json))

            # spacing unit
            unit = data.get("spacing_unit", None)
            unit_counter_json[unit] += 1
            unit_examples_json.setdefault(unit, anno.name)

            landmarks = data["landmarks"]
        except Exception as e:
            print(f"[WARN] Could not read JSON {anno.name}: {e}")
            continue

        # --- Landmark validity check ---
        valid_count = 0
        for lname, coords in landmarks.items():
            if is_missing(coords):
                cases_missing_landmarks[case].append((lname, coords))
            else:
                valid_count += 1
        landmark_count_hist[valid_count] += 1

    # ---- Reporting ----
    total_labeled = sum(landmark_count_hist.values())

    print(f"\nTotal volumes: {len(volumes)}")
    print(f"Total annotations: {len(labels)}")
    print(f"Cases with both image and label: {total_labeled}")
    print(f"Images without annotation: {len(images_without_label)}")
    print(f"Annotations without image: {len(labels_without_image)}")

    if images_without_label:
        print("\nImages without annotation:")
        for c in images_without_label:
            print(" ", c)
    if labels_without_image:
        print("\nAnnotations without image:")
        for c in labels_without_image:
            print(" ", c)

    print("\nHistogram: number of valid annotated landmarks per case:")
    for n, cnt in sorted(landmark_count_hist.items()):
        print(f"  {n} landmarks: {cnt} cases")
    if cases_missing_landmarks:
        print("\nCases with missing/invalid landmarks:")
        for case, probs in cases_missing_landmarks.items():
            for lname, coords in probs:
                print(f"  {case} | {lname}: {coords}")

    print("\nHistogram of NIfTI voxel spacings:")
    for sp, cnt in spacing_counter_nifti.items():
        print(f"  {sp}: {cnt} cases (e.g. {spacing_examples_nifti[sp]})")

    print("\nHistogram of JSON spacing values:")
    for sp, cnt in spacing_counter_json.items():
        print(f"  {sp}: {cnt} cases (e.g. {spacing_examples_json[sp]})")

    print("\nHistogram of JSON spacing units:")
    for unit, cnt in unit_counter_json.items():
        print(f"  {unit}: {cnt} cases (e.g. {unit_examples_json[unit]})")
        if unit != "mm":
            print(f"    >> WARNING: non-mm unit detected ({unit_examples_json[unit]})")

    print("\nNIfTI orientation codes (e.g. RAS, LPI):")
    for code, cnt in orientation_counter.items():
        print(f"  {code}: {cnt} cases (e.g. {orientation_examples[code]})")

    print("\nCases with spacing mismatches (NIfTI vs JSON):")
    print(len(spacing_mismatches))
    # if spacing_mismatches:
    #     for case, nif, js in spacing_mismatches:
    #         print(f"  {case}: NIfTI {nif} vs JSON {js}")
    # else:
    #     print("  None")

if __name__ == "__main__":
    main()

