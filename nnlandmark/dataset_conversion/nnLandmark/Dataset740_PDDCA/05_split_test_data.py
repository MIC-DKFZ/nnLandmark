#!/usr/bin/env python3
import argparse
import shutil
import random
from pathlib import Path

# ---------- helpers ----------
def strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"):    return name[:-4]
    if name.endswith(".nrrd"):   return name[:-5]
    if name.endswith(".nhdr"):   return name[:-5]
    return Path(name).stem

def strip_modality(stem: str) -> str:
    # remove trailing '_0000' if present (images)
    return stem[:-5] if stem.endswith("_0000") else stem

def canonical_key_from_name(name: str) -> str:
    """
    Produce canonical case key for PDDCA files.
    Examples:
      'caseA_0000.nrrd' -> 'caseA'
      'caseA.nrrd'      -> 'caseA'
      'caseA_0000.nhdr' -> 'caseA'
    """
    base = strip_ext(name)
    return strip_modality(base)

def scan_pairs(images_dir: Path, labels_dir: Path) -> dict[str, tuple[Path, Path]]:
    """
    Build {case_key: (img_path, lbl_path)} including only cases where BOTH image & label exist.
    images_dir contains files like {case}_0000.nrrd, labels_dir contains {case}.nrrd (or .nhdr).
    """
    imgs: dict[str, Path] = {}
    labels: dict[str, Path] = {}

    if images_dir.exists():
        for p in images_dir.iterdir():
            if p.is_file():
                imgs[canonical_key_from_name(p.name)] = p

    if labels_dir.exists():
        for p in labels_dir.iterdir():
            if p.is_file():
                labels[canonical_key_from_name(p.name)] = p

    pairs: dict[str, tuple[Path, Path]] = {}
    for k, ip in imgs.items():
        lp = labels.get(k)
        if lp is not None:
            pairs[k] = (ip, lp)
    return pairs

def ensure_dirs(root: Path):
    (root / "imagesTs").mkdir(parents=True, exist_ok=True)
    (root / "labelsTs").mkdir(parents=True, exist_ok=True)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Select N random PDDCA cases and move/copy image+label into imagesTs/labelsTs")
    ap.add_argument("--root",
        default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA",
        help="PDDCA dataset root (contains imagesTr/ and labelsTr/)")
    ap.add_argument("--n_test", type=int, default=7, help="Number of random test cases to select")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing target files")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"

    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"imagesTr or labelsTr missing under {root}")

    pairs = scan_pairs(images_dir, labels_dir)
    keys = sorted(pairs.keys())
    if not keys:
        raise SystemExit("No paired cases found in imagesTr/labelsTr")

    rng = random.Random(args.seed)
    n = min(args.n_test, len(keys))
    test_keys = sorted(rng.sample(keys, k=n))
    train_keys = sorted(k for k in keys if k not in test_keys)

    ensure_dirs(root)

    op = shutil.copy2 if args.copy else shutil.move

    def put(src: Path, dst: Path):
        if dst.exists():
            if args.overwrite:
                dst.unlink()
            else:
                return
        op(str(src), str(dst))

    # move/copy test keys to imagesTs/labelsTs
    for k in test_keys:
        img, lbl = pairs[k]
        put(img, root / "imagesTs" / img.name)
        put(lbl, root / "labelsTs" / lbl.name)

    # remaining go to imagesTr/labelsTr (they may already be there; if moved above they won't exist)
    for k in train_keys:
        img, lbl = pairs[k]
        # if original file was moved to imagesTs, it won't exist; skip in that case
        if img.exists():
            put(img, root / "imagesTr" / img.name)
        if lbl.exists():
            put(lbl, root / "labelsTr" / lbl.name)

    print(f"Selected {len(test_keys)} test cases: {test_keys}")
    print(f"Training cases: {len(train_keys)}")

if __name__ == "__main__":
    main()