#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round all voxel coordinates in all_landmarks_voxel.json to integers.

Expected input structure:
{
  "<case_id>": {
    "landmark_1": [x, y, z],
    "landmark_2": [x, y, z],
    ...
  },
  ...
}

Usage:
  python round_all_landmarks_voxel.py \
      --in_json /path/to/all_landmarks_voxel.json \
      --out_json /path/to/all_landmarks_voxel_int.json
  # (Pass the same path for --out_json to overwrite in place.)
"""

import os
import json
import argparse
import numpy as np


def load_json(fp: str):
    with open(fp, "r") as f:
        return json.load(f)


def save_json(obj, fp: str):
    os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)


def round_triplet(xyz):
    arr = np.asarray(xyz, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3 elements for coordinates, got shape {arr.shape} with value {xyz}")
    return np.round(arr).astype(int).tolist()  # nearest-integer rounding


def convert(in_json: str, out_json: str):
    data = load_json(in_json)

    # Validate top-level is a dict of cases
    if not isinstance(data, dict):
        raise TypeError("Top-level JSON must be a dict: {case_id: {...}}")

    out = {}
    for case_id, landmarks in data.items():
        if not isinstance(landmarks, dict):
            # If unexpected, copy through unchanged
            out[case_id] = landmarks
            continue

        new_landmarks = {}
        for lname, coords in landmarks.items():
            try:
                new_landmarks[lname] = round_triplet(coords)
            except Exception as e:
                # Keep original entry if malformed
                print(f"[WARN] {case_id} {lname}: {e}. Keeping original value.")
                new_landmarks[lname] = coords
        out[case_id] = new_landmarks

    save_json(out, out_json)
    print(f"Saved integer-rounded landmarks to: {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", default='/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset732_Afids/all_landmarks_voxel_float.json',
                    help="Path to all_landmarks_voxel.json")
    ap.add_argument("--out_json", default='/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset732_Afids/all_landmarks_voxel.json',
                    help="Path to save the integer-rounded JSON")
    args = ap.parse_args()

    convert(args.in_json, args.out_json)


if __name__ == "__main__":
    main()
