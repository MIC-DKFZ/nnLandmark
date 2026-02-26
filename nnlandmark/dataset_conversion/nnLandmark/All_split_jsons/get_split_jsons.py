#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def case_id_from_fname(p: Path) -> str:
    stem = p.name
    # remove common extensions
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    # remove trailing modality index like "_0000"
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem

def collect_case_ids(images_dir: Path) -> list:
    if not images_dir.exists():
        return []
    ids = set()
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        ids.add(case_id_from_fname(p))
    return sorted(ids)

def main():
    ap = argparse.ArgumentParser(description="Generate JSON with case ids from imagesTr/imagesTs (strip _0000).")
    ap.add_argument("--dataset_root", default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4", help="Path to nnLM_raw dataset folder (contains imagesTr/imagesTs)")
    ap.add_argument("--out", "-o", default="/home/a332l/src/nnunet-regression-dice/nnunetv2/dataset_conversion/nnLandmark/All_split_jsons/split_Dataset742_FeTA_2_4.json", help="Output JSON path")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    tr = collect_case_ids(root / "imagesTr")
    ts = collect_case_ids(root / "imagesTs")

    out = {"imagesTr": tr, "imagesTs": ts, "all": sorted(set(tr) | set(ts))}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out} (Tr={len(tr)} Ts={len(ts)})")

if __name__ == "__main__":
    main()