#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Copy & rename case .nrrd images into imagesTr folder as {case}_0000.<ext>"
    )
    parser.add_argument(
        "--base",
        default="/path/to/2024_Ertl_nnLandmark/data/PDDCA/PDDCA-1.4.1",
        help="Root containing case subdirectories"
    )
    parser.add_argument(
        "--target",
        default="/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA/imagesTr",
        help="Target imagesTr directory"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in target"
    )
    args = parser.parse_args()

    base = Path(os.path.expanduser(args.base))
    target = Path(os.path.expanduser(args.target))
    target.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        raise SystemExit(f"Base path does not exist: {base}")

    copied = 0
    skipped = 0
    missing = []

    for case_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        # prefer img.nrrd / image.nrrd or any .nrrd/.nhdr in the case dir root
        candidates = []
        for name in ("img.nrrd", "image.nrrd", "img.nhdr", "image.nhdr"):
            p = case_dir / name
            if p.exists():
                candidates.append(p)
        if not candidates:
            # fallback: any .nrrd or .nhdr file in the case dir (non-recursive)
            candidates = sorted(case_dir.glob("*.nrrd")) + sorted(case_dir.glob("*.nhdr"))

        if not candidates:
            missing.append(case_dir.name)
            skipped += 1
            print(f"[WARN] No .nrrd/.nhdr found in {case_dir.name}, skipping")
            continue

        src = candidates[0]
        # preserve extension; name target as {case}_0000.<ext>
        ext = "".join(src.suffixes)  # handles .nii.gz style too (though here .nhdr/.nrrd)
        dst_name = f"{case_dir.name}_0000{ext}"
        dst = target / dst_name

        if dst.exists() and not args.overwrite:
            print(f"[SKIP] Target exists and overwrite not set: {dst_name}")
            skipped += 1
            continue

        try:
            shutil.copy2(src, dst)
            copied += 1
            print(f"Copied {case_dir.name}: {src.name} -> {dst_name}")
        except Exception as e:
            print(f"[ERROR] Failed to copy {src} -> {dst}: {e}")
            skipped += 1

    print(f"\nSummary: copied={copied}, skipped={skipped}, missing_cases={len(missing)}")
    if missing:
        print("Missing images for cases (sample):", missing[:20])

if __name__ == "__main__":
    main()
