#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path

# ---------- helpers ----------
def strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"):    return name[:-4]
    return Path(name).stem

def strip_modality(stem: str) -> str:
    # remove trailing '_0000' if present (images)
    return stem[:-5] if stem.endswith("_0000") else stem

def canonical_key_from_name(name: str) -> str:
    """
    Canonical case key: start at first 'sub-' if present, then drop '_0000' and extension.
    Works for:
      'AFIDs-HCP_sub-103111_0000.nii.gz' -> 'sub-103111'
      'sub-103111_0000.nii.gz'           -> 'sub-103111'
      'AFIDs-HCP_sub-103111.nii.gz'      -> 'sub-103111'
      'sub-103111.nii.gz'                -> 'sub-103111'
    """
    base = strip_ext(name)
    i = base.find("sub-")
    if i >= 0:
        base = base[i:]
    return strip_modality(base)

def scan_pairs(images_dir: Path, labels_dir: Path) -> dict[str, tuple[Path, Path]]:
    """
    Build {case_key: (img_path, lbl_path)} for a directory pair.
    Only includes cases where BOTH image & label exist.
    """
    imgs = {}
    for p in images_dir.glob("*"):
        if p.is_file():
            key = canonical_key_from_name(p.name)
            imgs[key] = p
    lbls = {}
    for p in labels_dir.glob("*"):
        if p.is_file():
            key = canonical_key_from_name(p.name)
            lbls[key] = p
    pairs = {}
    for k, ip in imgs.items():
        lp = lbls.get(k)
        if lp is not None:
            pairs[k] = (ip, lp)
    return pairs

def ensure_dirs(root: Path, split: str):
    (root / f"images{split}").mkdir(parents=True, exist_ok=True)
    (root / f"labels{split}").mkdir(parents=True, exist_ok=True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Reuse old Ts/Tr split by scanning old imagesTs/labelsTs.")
    ap.add_argument("--new_root",
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset732_Afids/",
        help="NEW dataset root; contains imagesAll/ and labelsAll/")
    ap.add_argument("--old_root",
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/first_submission/nnUNet_raw/Dataset704_afids/",
        help="OLD dataset root; contains imagesTs/, labelsTs/ (and typically imagesTr/, labelsTr/)")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    new_root = Path(args.new_root)
    old_root = Path(args.old_root)

    images_all = new_root / "imagesAll"
    labels_all = new_root / "labelsAll"
    if not images_all.is_dir() or not labels_all.is_dir():
        raise SystemExit(f"❌ imagesAll or labelsAll not found under {new_root}")

    # 1) Define TEST SET from OLD dataset by intersecting imagesTs & labelsTs (guaranteed pairs).
    old_ts_images = old_root / "imagesTs"
    old_ts_labels = old_root / "labelsTs"
    if not old_ts_images.is_dir() or not old_ts_labels.is_dir():
        raise SystemExit(f"❌ imagesTs or labelsTs not found under {old_root}")
    old_ts_pairs = scan_pairs(old_ts_images, old_ts_labels)  # keys are like 'sub-103111'
    ts_keys = set(old_ts_pairs.keys())
    if not ts_keys:
        raise SystemExit("❌ No paired cases found in old imagesTs/labelsTs.")

    # 2) Collect available pairs in NEW dataset (imagesAll & labelsAll).
    new_pairs = scan_pairs(images_all, labels_all)  # {key: (img, lbl)}
    new_keys = set(new_pairs.keys())

    # 3) Split: Ts = intersection with old Ts; Tr = remaining paired cases.
    ts_keys_new = sorted(ts_keys & new_keys)
    missing_in_new = sorted(ts_keys - new_keys)
    tr_keys_new = sorted(new_keys - set(ts_keys_new))

    # 4) Prepare output dirs.
    ensure_dirs(new_root, "Ts")
    ensure_dirs(new_root, "Tr")
    op = shutil.move if args.move else shutil.copy

    # 5) Copy/move Ts.
    n_ts = 0
    for k in ts_keys_new:
        img_src, lbl_src = new_pairs[k]
        op(img_src, new_root / "imagesTs" / img_src.name)
        op(lbl_src, new_root / "labelsTs" / lbl_src.name)
        n_ts += 1

    # 6) Copy/move Tr (all remaining paired cases).
    n_tr = 0
    for k in tr_keys_new:
        img_src, lbl_src = new_pairs[k]
        op(img_src, new_root / "imagesTr" / img_src.name)
        op(lbl_src, new_root / "labelsTr" / lbl_src.name)
        n_tr += 1

    # 7) Report.
    print(f"✅ Ts: {n_ts} pairs copied from new imagesAll/labelsAll (based on old Ts).")
    print(f"✅ Tr: {n_tr} pairs copied (remaining paired cases).")

    if missing_in_new:
        print("\n⚠️  Old test cases missing in NEW dataset (no image+label pair found):")
        for k in missing_in_new[:50]:
            print("   -", k)
        if len(missing_in_new) > 50:
            print(f"   … and {len(missing_in_new) - 50} more")

    # 8) Final on-disk check.
    cnt = lambda p: len(list(p.glob("*")))
    print("\nOn disk:")
    print(f"  imagesTs={cnt(new_root/'imagesTs')}, labelsTs={cnt(new_root/'labelsTs')}")
    print(f"  imagesTr={cnt(new_root/'imagesTr')}, labelsTr={cnt(new_root/'labelsTr')}")

if __name__ == "__main__":
    main()

