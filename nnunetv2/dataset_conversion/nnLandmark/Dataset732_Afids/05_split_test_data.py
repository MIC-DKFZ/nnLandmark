#!/usr/bin/env python3
import argparse, shutil, random
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
    ap = argparse.ArgumentParser(description="Create a random Tr/Ts split from imagesAll/labelsAll.")
    ap.add_argument("--new_root",
        default="/path/to/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset732_Afids/",
        help="NEW dataset root; contains imagesAll/ and labelsAll/")
    ap.add_argument("--test-size", "-t", type=float, default=0.2,
        help="If <1 treat as fraction of cases to use for test set; if >=1 treat as integer number of test cases.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    new_root = Path(args.new_root)
    images_all = new_root / "imagesAll"
    labels_all = new_root / "labelsAll"
    if not images_all.is_dir() or not labels_all.is_dir():
        raise SystemExit(f"❌ imagesAll or labelsAll not found under {new_root}")

    # Build paired cases
    pairs = scan_pairs(images_all, labels_all)  # {key: (img, lbl)}
    keys = sorted(pairs.keys())
    n = len(keys)
    if n == 0:
        raise SystemExit("❌ No paired cases found in imagesAll/labelsAll.")

    # Determine number of test cases
    ts_count = args.test_size
    if ts_count < 1:
        n_ts = int(round(n * ts_count))
    else:
        n_ts = int(ts_count)
    n_ts = max(1, min(n - 1, n_ts))  # ensure at least 1 test and at least 1 train

    # Random sampling
    rng = random.Random(args.seed)
    ts_keys_new = sorted(rng.sample(keys, n_ts))
    tr_keys_new = sorted(k for k in keys if k not in ts_keys_new)

    # Prepare output dirs.
    ensure_dirs(new_root, "Ts")
    ensure_dirs(new_root, "Tr")
    op = shutil.move if args.move else shutil.copy

    # Copy/move Ts.
    n_ts_done = 0
    for k in ts_keys_new:
        img_src, lbl_src = pairs[k]
        op(img_src, new_root / "imagesTs" / img_src.name)
        op(lbl_src, new_root / "labelsTs" / lbl_src.name)
        n_ts_done += 1

    # Copy/move Tr (all remaining paired cases).
    n_tr_done = 0
    for k in tr_keys_new:
        img_src, lbl_src = pairs[k]
        op(img_src, new_root / "imagesTr" / img_src.name)
        op(lbl_src, new_root / "labelsTr" / lbl_src.name)
        n_tr_done += 1

    # Report.
    print(f"✅ Random split created with seed={args.seed}")
    print(f"  Total paired cases: {n}")
    print(f"  Ts: {n_ts_done} pairs -> {new_root/'imagesTs'}")
    print(f"  Tr: {n_tr_done} pairs -> {new_root/'imagesTr'}")

    # Final on-disk check.
    cnt = lambda p: len(list(p.glob("*")))
    print("\nOn disk:")
    print(f"  imagesTs={cnt(new_root/'imagesTs')}, labelsTs={cnt(new_root/'labelsTs')}")
    print(f"  imagesTr={cnt(new_root/'imagesTr')}, labelsTr={cnt(new_root/'labelsTr')}")

if __name__ == "__main__":
    main()
