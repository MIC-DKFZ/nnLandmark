#!/usr/bin/env python3
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image


# ---------- helpers ----------
def is_missing_xy(xy):
    """
    xy: iterable of [x, y]
    Missing if: any NaN, all zeros, all -1, or any negative.
    """
    x, y = xy
    vals = [x, y]
    if any(pd.isna(v) for v in vals):
        return True
    if all(np.isclose(v, 0) for v in vals):
        return True
    if all(np.isclose(v, -1) for v in vals):
        return True
    if any(v < 0 for v in vals):
        return True
    return False


def detect_landmark_ids(columns):
    """
    Columns expected like: p1x, p1y, ..., p53x, p53y
    Returns sorted list of integer landmark ids present.
    """
    ids = set()
    for c in columns:
        m = re.match(r"p(\d+)[xy]$", c)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


# ---------- main ----------
def main():
    # ---- Paths ----
    images_dir = Path("/home/a332l/dev/Project_nnLandmark/data/CL-Detection2024_Accessible_Data/Training_Set/images/")
    labels_csv = Path("/home/a332l/dev/Project_nnLandmark/data/CL-Detection2024_Accessible_Data/Training_Set/labels.csv")

    # ---- Load CSV ----
    df = pd.read_csv(labels_csv)
    if "image file" not in df.columns:
        raise RuntimeError("CSV is expected to have a column named 'image file'.")

    # landmark id detection
    lm_ids = detect_landmark_ids(df.columns)

    # ---- Files on disk vs CSV ----
    # consider bmp, BMP, png, jpg just in case (mostly bmp expected)
    disk_images = {p.name for p in sum(
        [list(images_dir.glob("*.bmp")),
         list(images_dir.glob("*.BMP")),
         list(images_dir.glob("*.png")),
         list(images_dir.glob("*.jpg")),
         list(images_dir.glob("*.jpeg"))], []
    )}
    csv_images = set(df["image file"].astype(str).tolist())

    images_without_label = sorted(disk_images - csv_images)
    labels_without_image = sorted(csv_images - disk_images)

    # ---- Counters & Collectors ----
    sizes_counter          = Counter()
    sizes_examples         = {}
    mode_counter           = Counter()
    mode_examples          = {}

    spacing_counter        = Counter()
    spacing_examples       = {}
    spacing_values         = []

    landmark_count_hist    = Counter()
    cases_missing_landmarks = defaultdict(list)

    # ---- Iterate rows (each = one case) ----
    for _, row in df.iterrows():
        img_name = str(row["image file"])
        img_path = images_dir / img_name
        if not img_path.exists():
            # accounted in labels_without_image already; skip deep analysis
            continue

        # image info
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                sizes_counter[(w, h)] += 1
                sizes_examples.setdefault((w, h), img_name)

                mode_counter[im.mode] += 1
                mode_examples.setdefault(im.mode, img_name)
        except Exception as e:
            print(f"[WARN] Could not open image {img_name}: {e}")
            continue

        # spacing (single isotropic value in mm/pixel expected)
        if "spacing(mm)" in df.columns:
            try:
                sp = float(row["spacing(mm)"])
                spacing_counter[(round(sp, 6),)] += 1  # keep as 1-tuple for consistent printing
                spacing_examples.setdefault((round(sp, 6),), img_name)
                spacing_values.append(sp)
            except Exception:
                pass

        # landmark validity check
        valid = 0
        for k in lm_ids:
            x = row.get(f"p{k}x", np.nan)
            y = row.get(f"p{k}y", np.nan)
            if is_missing_xy([x, y]):
                cases_missing_landmarks[img_name].append((f"p{k}", [x, y]))
            else:
                valid += 1
        landmark_count_hist[valid] += 1

    # ---- Reporting ----
    print(f"\nTotal images on disk: {len(disk_images)}")
    print(f"Total rows in CSV:    {len(csv_images)}")
    print(f"Images referenced in CSV but missing on disk: {len(labels_without_image)}")
    print(f"Images on disk but not referenced in CSV:     {len(images_without_label)}")

    if labels_without_image:
        print("\nReferenced in CSV but missing on disk:")
        for n in labels_without_image:
            print(" ", n)

    if images_without_label:
        print("\nOn disk but not referenced in CSV:")
        for n in images_without_label:
            print(" ", n)

    print("\nHistogram: number of valid annotated landmarks per image:")
    for n, cnt in sorted(landmark_count_hist.items()):
        print(f"  {n} landmarks: {cnt} images")

    if cases_missing_landmarks:
        print("\nImages with missing/invalid landmarks:")
        for img_name, probs in cases_missing_landmarks.items():
            for lname, coords in probs:
                print(f"  {img_name} | {lname}: {coords}")

    print("\nHistogram of image sizes (W x H):")
    for (w, h), cnt in sizes_counter.items():
        print(f"  {w}x{h}: {cnt} images (e.g. {sizes_examples[(w, h)]})")

    print("\nHistogram of image modes:")
    for m, cnt in mode_counter.items():
        print(f"  {m}: {cnt} images (e.g. {mode_examples[m]})")

    if spacing_counter:
        print("\nHistogram of spacing(mm) values:")
        for sp, cnt in spacing_counter.items():
            print(f"  {sp}: {cnt} images (e.g. {spacing_examples[sp]})")

        arr = np.array(spacing_values, dtype=float)
        print("\nSpacing(mm) summary:")
        print(f"  min: {np.min(arr):.6f}")
        print(f"  max: {np.max(arr):.6f}")
        print(f"  mean: {np.mean(arr):.6f}")
        print(f"  median: {np.median(arr):.6f}")
    else:
        print("\nNo 'spacing(mm)' column found or parseable in CSV.")

    # Optional: write a compact JSON summary next to the CSV for later reference
    out_summary = {
        "sizes_counts": {f"{w}x{h}": cnt for (w, h), cnt in sizes_counter.items()},
        "modes_counts": dict(mode_counter),
        "images_without_label": images_without_label,
        "labels_without_image": labels_without_image,
        "landmarks_per_image_hist": dict(landmark_count_hist),
        "spacing_hist": {str(sp): cnt for sp, cnt in spacing_counter.items()},
    }
    with open(labels_csv.parent / "analysis_summary.json", "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nWrote summary JSON → {labels_csv.parent / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()

