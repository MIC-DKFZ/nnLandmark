#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Set, Dict, Any

KEEP = {
    "landmark_3_1",
    "landmark_3_2",
    "landmark_5_1",
    "landmark_5_2",
    "landmark_6_1",
    "landmark_6_2",
}


def filter_landmarks(src: Path, dst: Path, keep: Set[str] = KEEP) -> None:
    with src.open("r") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}
    kept_cases = 0
    total_cases = 0
    total_kept_points = 0

    for case_id, lm_map in data.items():
        total_cases += 1
        if not isinstance(lm_map, dict):
            continue
        filtered = {k: v for k, v in lm_map.items() if k in keep}
        if filtered:
            out[case_id] = filtered
            kept_cases += 1
            total_kept_points += len(filtered)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Processed {total_cases} cases -> kept {kept_cases} cases ({total_kept_points} landmarks).")
    print(f"Wrote filtered JSON to: {dst}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Filter all_landmarks_voxel.json to keep only landmarks 3_*, 5_*, 6_*")
    p.add_argument("--src", type=Path, default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset741_LFC_cerebellum/all_landmarks_voxel.json"))
    p.add_argument("--out", type=Path, default=None, help="output path (default: same dir with suffix _3_5_6.json)")
    args = p.parse_args()

    src = args.src
    if args.out:
        dst = args.out
    else:
        dst = src.with_name(src.stem + "_3_5_6" + src.suffix)

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    filter_landmarks(src, dst)