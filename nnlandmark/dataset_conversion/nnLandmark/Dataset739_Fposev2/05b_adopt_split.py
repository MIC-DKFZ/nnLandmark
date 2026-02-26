#!/usr/bin/env python3
# filepath: nnunetv2/dataset_conversion/nnLandmark/Dataset739_Fposev2/06_apply_split.py
"""
Apply a provided split JSON (pose_dataset_split.json) to arrange images/labels into
imagesTr/imagesTs and labelsTr/labelsTs.

Example:
python 06_apply_split.py \
  --split /path/to/pose_dataset_split.json \
  --cross cross_1 \
  --data-root /path/to/nnUNet_raw/Dataset739_Fposev2 \
  --move --force

Defaults copy files (safe). Use --move to move instead.
"""
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = [".nii.gz", ".nii"]

def parse_case_name(name: str) -> str:
    # split entries look like "flp_0810_262_landmark.json"
    # strip common suffixes and extensions to get case id
    s = name
    if s.endswith(".json"):
        s = s[:-5]
    # remove trailing "_landmark" if present
    if s.endswith("_landmark"):
        s = s[:-9]
    return s

def find_file_for_case(folder: Path, case: str, exts: List[str]=IMAGE_EXTS) -> Optional[Path]:
    # try common patterns in order
    # 1) case_0000.ext
    for ext in exts:
        p = folder / f"{case}_0000{ext}"
        if p.exists():
            return p
    # 2) any file starting with case_ (case_*.nii*)
    for ext in exts:
        candidates = sorted(folder.glob(f"{case}_*{ext}"))
        if candidates:
            return candidates[0]
    # 3) exact case.ext
    for ext in exts:
        p = folder / f"{case}{ext}"
        if p.exists():
            return p
    # 4) any file whose stem equals case (covers other extensions)
    for p in folder.iterdir():
        if p.is_file():
            stem = p.name
            # remove .nii or .nii.gz
            if stem.endswith(".nii.gz"):
                stem_stem = stem[:-7]
            elif stem.endswith(".nii"):
                stem_stem = stem[:-4]
            else:
                stem_stem = Path(stem).stem
            if stem_stem == case:
                return p
    return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_move(src: Path, dst: Path, move: bool=False, force: bool=False, dry: bool=False) -> str:
    if not src.exists():
        return f"MISSING_SRC {src}"
    ensure_dir(dst.parent)
    if dst.exists():
        if not force:
            return f"SKIP_EXISTS {dst}"
        # else remove existing file
        if not dry:
            dst.unlink()
    if dry:
        return f"DRY {'move' if move else 'copy'} {src.name} -> {dst}"
    if move:
        shutil.move(str(src), str(dst))
        return f"MOVED {src.name} -> {dst.relative_to(dst.parent.parent)}"
    else:
        shutil.copy2(str(src), str(dst))
        return f"COPIED {src.name} -> {dst.relative_to(dst.parent.parent)}"

def apply_split(split_json: Path, cross: str, data_root: Path, move: bool=False, force: bool=False, dry: bool=False):
    s = json.loads(split_json.read_text())
    if cross not in s:
        raise SystemExit(f"Cross '{cross}' not found in split JSON. Available: {list(s.keys())}")

    entries = s[cross]
    test_list = entries.get("test", [])
    train_list = entries.get("train", [])
    val_list = entries.get("val", [])

    # val goes into train
    train_plus_val = train_list + val_list

    images_all = data_root / "imagesAll"
    labels_all = data_root / "labelsAll"
    images_tr = data_root / "imagesTr"
    images_ts = data_root / "imagesTs"
    labels_tr = data_root / "labelsTr"
    labels_ts = data_root / "labelsTs"

    for d in (images_tr, images_ts, labels_tr, labels_ts):
        ensure_dir(d)

    summary = {"copied":0, "moved":0, "skipped_exists":0, "missing_src":0, "errors":0, "dry": dry}

    def process_case_list(case_names: List[str], target_images_dir: Path, target_labels_dir: Path, which: str):
        for nm in case_names:
            case = parse_case_name(nm)
            # find image in imagesAll first; if not found, try in dataset root imagesTr/imagesTs (maybe already placed)
            src_img = find_file_for_case(images_all, case)
            if src_img is None:
                # try dataset root images (maybe user already has imagesTr/imagesTs)
                src_img = find_file_for_case(data_root, case)
            if src_img is None:
                summary["missing_src"] += 1
                print(f"[MISSING] image for case '{case}' not found (from {nm})")
            else:
                dst_img = target_images_dir / src_img.name
                try:
                    msg = copy_or_move(src_img, dst_img, move=move, force=force, dry=dry)
                    print(msg)
                    if msg.startswith("COPIED"):
                        summary["copied"] += 1
                    elif msg.startswith("MOVED"):
                        summary["moved"] += 1
                    elif msg.startswith("SKIP_EXISTS"):
                        summary["skipped_exists"] += 1
                    elif msg.startswith("MISSING_SRC"):
                        summary["missing_src"] += 1
                except Exception as e:
                    summary["errors"] += 1
                    print(f"[ERROR] copying/moving {src_img} -> {dst_img}: {e}")

            # labels: try to find label in labelsAll
            if labels_all.exists():
                # common label names: case.nii.gz or case.nii
                label_src = find_file_for_case(labels_all, case)
                if label_src is None:
                    # maybe label already in labelsTr/labelsTs - skip silently
                    print(f"[MISSING] label for case '{case}' not found in labelsAll")
                else:
                    dst_label = target_labels_dir / label_src.name
                    try:
                        msg = copy_or_move(label_src, dst_label, move=move, force=force, dry=dry)
                        print(msg)
                    except Exception as e:
                        summary["errors"] += 1
                        print(f"[ERROR] copying/moving label {label_src} -> {dst_label}: {e}")

    print(f"Applying split '{cross}': {len(test_list)} test cases -> imagesTs, {len(train_plus_val)} train cases -> imagesTr")
    process_case_list(test_list, images_ts, labels_ts, "test")
    process_case_list(train_plus_val, images_tr, labels_tr, "train+val")

    print("Summary:", summary)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/data/landmark_datasets_Shenzhen/pose_dataset/pose_dataset_split.json", help="Path to pose_dataset_split.json")
    ap.add_argument("--cross", type=str, default="cross_1", help="Which cross to use (e.g. cross_1)")
    ap.add_argument("--data-root", type=Path, default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2", help="nnLM_raw dataset root containing imagesAll/labelsAll")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files in target")
    ap.add_argument("--dry-run", action="store_true", help="Do not write, only show actions")
    args = ap.parse_args()

    apply_split(args.split, args.cross, args.data_root, move=args.move, force=args.force, dry=args.dry_run)

if __name__ == "__main__":
    main()