#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def is_missing_xy(xy):
    x, y = xy
    if pd.isna(x) or pd.isna(y):
        return True
    if all(np.isclose(v, 0) for v in (x, y)):
        return True
    if all(np.isclose(v, -1) for v in (x, y)):
        return True
    if x < 0 or y < 0:
        return True
    return False


def detect_landmark_ids(columns):
    """
    Finds landmark columns of the form p{idx}x / p{idx}y.
    Returns sorted list of integer landmark ids present.
    """
    ids = set()
    for c in columns:
        m = re.match(r"p(\d+)[xy]$", c)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def build_name_to_label(ordered_names):
    """
    ordered_names: list of landmark names in first-seen order (e.g., ["p1", "p2", ...]).
    If a name ends with digits, use that number as the label id; otherwise assign sequentially.
    """
    name_to_label = {}
    next_free = 1
    for name in ordered_names:
        if name in name_to_label:
            continue
        tail = name.split("_")[-1]  # handle "landmark_12" too
        if tail.isdigit():
            name_to_label[name] = int(tail)
        else:
            name_to_label[name] = next_free
            next_free += 1
    return name_to_label


def main():
    ap = argparse.ArgumentParser(description="Create 2D landmark JSONs from CL-Detection CSV.")
    ap.add_argument("--labels_csv", type=Path, default="/home/a332l/dev/Project_nnLandmark/data/CL-Detection2024_Accessible_Data/Training_Set/labels.csv",
                    help="CSV with columns: 'image file', 'spacing(mm)', p1x,p1y,...")
    ap.add_argument("--out_root", type=Path, default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/",
                    help="Output directory (e.g., nnLM_raw/Dataset9xx_CLD2D)")
    ap.add_argument("--prefix", default="",
                    help="Prefix for case ids (default: CLD2D_)")
    ap.add_argument("--coordinates_filename", default="landmark_coordinates.json",
                    help="Filename for coordinates JSON (default: landmark_coordinates.json)")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.labels_csv)
    if "image file" not in df.columns:
        raise RuntimeError("CSV is expected to have a column named 'image file'.")
    if "spacing(mm)" not in df.columns:
        print("[WARN] No 'spacing(mm)' column found. Will write spacing=1.0 mm isotropic.")
        df["spacing(mm)"] = 1.0

    # Landmark id detection
    lm_ids = detect_landmark_ids(df.columns)

    # Outputs
    coords_out = {}       # {case_id: {name: [x,y], ...}}
    spacing_out = {}      # {case_id: {"image_spacing":[sx,sy], "annotation_spacing": sx}}
    first_seen_names = [] # to build name_to_label with preserved order

    for _, row in df.iterrows():
        img_name = str(row["image file"])
        stem = Path(img_name).stem
        case_id = f"{args.prefix}{stem}"

        # spacing (assume isotropic)
        try:
            sp = float(row["spacing(mm)"])
        except Exception:
            sp = 1.0

        spacing_out[case_id] = {
            "image_spacing": [sp, sp],        # isotropic pixels: x,y
            "annotation_spacing": sp          # single scalar as in your previous 3D script
        }

        # landmarks
        lmks = {}
        for k in lm_ids:
            name = f"p{k}"
            x = row.get(f"p{k}x", np.nan)
            y = row.get(f"p{k}y", np.nan)
            if not is_missing_xy([x, y]):
                lmks[name] = [float(x), float(y)]
                first_seen_names.append(name)
        coords_out[case_id] = lmks

    # name_to_label with order preserved (first appearance wins)
    # Use unique order while preserving order:
    seen = set()
    ordered_unique_names = []
    for n in first_seen_names:
        if n not in seen:
            seen.add(n)
            ordered_unique_names.append(n)
    name_to_label = build_name_to_label(ordered_unique_names)

    # Write files
    (args.out_root / args.coordinates_filename).write_text(json.dumps(coords_out, indent=2))
    (args.out_root / "spacing.json").write_text(json.dumps(spacing_out, indent=2))
    (args.out_root / "name_to_label.json").write_text(json.dumps(name_to_label, indent=2))

    print(f"✅ wrote {args.coordinates_filename}, spacing.json, and name_to_label.json to {args.out_root}")
    print(f"   cases: {len(coords_out)} | landmarks observed: {len(name_to_label)}")


if __name__ == "__main__":
    main()
