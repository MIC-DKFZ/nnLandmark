# Keep only cases with all 14 landmarks annotated

#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# ---------- name handling ----------
def strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"):    return name[:-4]
    if name.endswith(".nrrd"):   return name[:-5]
    return Path(name).stem

def strip_modality(stem: str) -> str:
    # remove trailing '_0000' modality tag if present
    return stem[:-5] if stem.endswith("_0000") else stem

def canonical_key(name: str) -> str:
    """
    Stable case key across old/new names.
    Prefer substring starting at 'sub-' if present, then drop '_0000' and extension.
    """
    base = strip_ext(name)
    i = base.find("sub-")
    if i >= 0:
        base = base[i:]
    return strip_modality(base)

# ---------- scanning ----------
def scan_dir(dir_path: Path) -> dict[str, Path]:
    """Return {case_key: path} for files in a dir."""
    out = {}
    if not dir_path.exists():
        return out
    for p in dir_path.iterdir():
        if p.is_file():
            out[canonical_key(p.name)] = p
    return out

# ---------- label check ----------
def has_all_landmarks(lbl_path: Path, required=set(range(1, 15))) -> bool:
    """Return True if label volume contains ALL ids in required (ignores 0)."""
    img = sitk.ReadImage(str(lbl_path))
    arr = sitk.GetArrayFromImage(img)
    present = set(np.unique(arr).tolist())
    present.discard(0)
    return required.issubset(present)

# ---------- deletion ----------
def rm(p: Path, apply: bool):
    if p.exists():
        if apply:
            p.unlink()
        return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Keep only cases with all 14 landmark labels; remove others from All/Tr/Ts.")
    ap.add_argument("--root", default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset733_MML_comp/",
                    help="nnUNet_raw dataset root containing imagesVal/labelsVal (+ Tr/Ts)")
    ap.add_argument("--n_labels", type=int, default=14, help="number of required landmark ids (1..n)")
    ap.add_argument("--apply", action="store_true", help="actually delete files (default: dry-run)")
    args = ap.parse_args()

    root = Path(args.root)

    # Folders (some may not exist; that’s fine)
    dirs = {
        "imagesVal": root / "imagesVal",
        "labelsVal": root / "labelsVal",
        "imagesTr":  root / "imagesTr",
        "labelsTr":  root / "labelsTr",
        "imagesTs":  root / "imagesTs",
        "labelsTs":  root / "labelsTs",
    }

    # Scan once
    maps = {k: scan_dir(v) for k, v in dirs.items()}

    # Decide which label file to use for each case:
    # prefer labelsVal, else from labelsTr, else labelsTs.
    label_sources = {}
    for key, p in maps["labelsVal"].items():
        label_sources[key] = p
    for key, p in maps["labelsTr"].items():
        label_sources.setdefault(key, p)
    for key, p in maps["labelsTs"].items():
        label_sources.setdefault(key, p)

    if not label_sources:
        raise SystemExit("❌ No label files found in labelsVal/labelsTr/labelsTs.")

    required = set(range(1, args.n_labels + 1))

    keep = set()
    drop = set()

    # Evaluate each case that has any label
    for key, lbl_path in sorted(label_sources.items()):
        ok = False
        try:
            ok = has_all_landmarks(lbl_path, required=required)
        except Exception as e:
            print(f"⚠️  Could not read label for {key}: {lbl_path} ({e})")
        if ok:
            keep.add(key)
        else:
            drop.add(key)

    # Also drop any case that has images but no label anywhere
    all_image_keys = set(maps["imagesVal"].keys()) | set(maps["imagesTr"].keys()) | set(maps["imagesTs"].keys())
    keys_without_label = all_image_keys - set(label_sources.keys())
    drop.update(keys_without_label)

    print(f"\nSummary (dry-run by default):")
    print(f"  Cases with ALL {args.n_labels} landmarks: {len(keep)} (kept)")
    print(f"  Cases missing landmarks or labels:       {len(drop)} (to remove)")
    if keys_without_label:
        print(f"    …of which without any label file:      {len(keys_without_label)}")

    # Show a sample
    for title, keys in (("Keep", sorted(list(keep))[:5]), ("Remove", sorted(list(drop))[:10])):
        if keys:
            print(f"  {title} sample:", ", ".join(keys))

    # Build list of files to delete
    to_delete = []
    for key in sorted(drop):
        for dname in ("imagesVal", "labelsVal", "imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            p = maps[dname].get(key)
            if p is not None:
                to_delete.append(p)

    print(f"\nWill {'DELETE' if args.apply else 'mark to delete'} {len(to_delete)} files across All/Tr/Ts.")
    if not args.apply:
        print("   (no files were deleted; re-run with --apply to proceed)")
        return

    # Apply deletion
    removed = 0
    for p in to_delete:
        if rm(p, apply=True):
            removed += 1
    print(f"✅ Deleted {removed} files. Done.")

if __name__ == "__main__":
    main()
