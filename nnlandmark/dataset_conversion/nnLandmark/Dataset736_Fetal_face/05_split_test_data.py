#!/usr/bin/env python3
"""
Randomly split <imagesAll> / <labelsAll> into
    imagesTs / labelsTs   (exactly N test cases)
    imagesTr / labelsTr   (the remaining cases)

Usage
-----
python split_train_test.py \
       --images  /path/imagesAll \
       --labels  /path/labelsAll \
       --outdir  /path            \
       --n_test  100              \
       [--seed 42]                \
       [--move]                   # move instead of copy
"""

import argparse, random, shutil
from pathlib import Path

def case_id_from_image(fn: Path) -> str:
    """Strip '_0000' and (double) extension."""
    stem = fn.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    else:  # .nrrd or others
        stem = fn.stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem

def collect_pairs(img_dir: Path, lbl_dir: Path):
    pairs = {}
    for img in img_dir.iterdir():
        case = case_id_from_image(img)
        lbl_matches = list(lbl_dir.glob(f"{case}.*"))
        if lbl_matches:
            pairs[case] = (img, lbl_matches[0])
    return pairs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset736_Fetal_face/imagesAll/", help="imagesAll directory")
    ap.add_argument("--labels", default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset736_Fetal_face/labelsAll/", help="labelsAll directory")
    ap.add_argument("--outdir", default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset736_Fetal_face/", help="folder where new dirs are created")
    ap.add_argument("--n_test", type=int, default=25, help="number of test cases")
    ap.add_argument("--seed",   type=int, default=42,  help="RNG seed for reproducibility")
    ap.add_argument("--move",   action="store_true",   help="move files instead of copying")
    args = ap.parse_args()

    random.seed(args.seed)

    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    pairs = collect_pairs(img_dir, lbl_dir)
    if len(pairs) < args.n_test:
        raise SystemExit(f"Only {len(pairs)} pairs found — cannot pick {args.n_test} test cases.")

    test_keys = random.sample(list(pairs), args.n_test)
    train_keys = [k for k in pairs if k not in test_keys]

    for split, keys in (("Ts", test_keys), ("Tr", train_keys)):
        (Path(args.outdir) / f"images{split}").mkdir(parents=True, exist_ok=True)
        (Path(args.outdir) / f"labels{split}").mkdir(parents=True, exist_ok=True)

        for k in keys:
            img_src, lbl_src = pairs[k]
            img_dst = Path(args.outdir) / f"images{split}" / img_src.name
            lbl_dst = Path(args.outdir) / f"labels{split}" / lbl_src.name
            op = shutil.move if args.move else shutil.copy
            op(img_src, img_dst)
            op(lbl_src, lbl_dst)

    print(f"✅  {args.n_test} cases → Ts, {len(train_keys)} cases → Tr")

if __name__ == "__main__":
    main()
