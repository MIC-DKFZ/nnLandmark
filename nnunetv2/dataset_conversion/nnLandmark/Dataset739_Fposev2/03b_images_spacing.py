#!/usr/bin/env python3
"""
Fix NIfTI image spacing (header pixdim + affine) using spacing.json entries.

Usage examples:
# dry-run, only print planned changes
python fix_image_spacing.py --images-root /path/to/nnUNet_raw/Dataset737_DMGLD_LFC/imagesTr --spacing /path/to/spacing.json --dry-run

# apply changes and back up originals
python fix_image_spacing.py --images-root /path/to/nnUNet_raw/Dataset737_DMGLD_LFC --spacing /path/to/spacing.json --backup
"""
from pathlib import Path
import json
import argparse
import shutil
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional, Tuple

def find_label_for_case(img_path: Path, case_id: str) -> Optional[Path]:
    """
    Search for a label file corresponding to case_id.
    Looks in sibling label directories commonly used in nnUNet: labelsTr, labelsTs, labels
    """
    parent = img_path.parent
    # If image is under imagesTr/imagesTs, dataset root is parent.parent
    dataset_root = parent.parent if parent.name.startswith("images") else parent
    candidate_dirs = [
        dataset_root / "labelsTr",
        dataset_root / "labelsTs",
        dataset_root / "labels",
        dataset_root / "labelsAll",
    ]
    for d in candidate_dirs:
        if not d.exists():
            continue
        for ext in (".nii.gz", ".nii"):
            p = d / f"{case_id}{ext}"
            if p.exists():
                return p
    # not found
    return None

def load_spacing_json(path: Path) -> Dict[str, Tuple[float, float, float]]:
    raw = json.loads(path.read_text())
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            # Prefer annotation_spacing, then image_spacing
            if "annotation_spacing" in v and v["annotation_spacing"] is not None:
                vals = v["annotation_spacing"]
            elif "image_spacing" in v and v["image_spacing"] is not None:
                vals = v["image_spacing"]
            elif "spacing" in v and v["spacing"] is not None:
                vals = v["spacing"]
            else:
                raise ValueError(f"Unrecognized spacing dict for case '{k}': {v}")
        elif isinstance(v, (list, tuple)) and len(v) == 3:
            vals = v
        else:
            raise ValueError(f"Unrecognized spacing format for case '{k}': {v}")
        out[k] = (float(vals[0]), float(vals[1]), float(vals[2]))
    return out

def iter_image_files(images_root: Path):
    # Accept images root that may itself be imagesTr/imagesTs or contain them
    if (images_root / "imagesTr").is_dir() or (images_root / "imagesTs").is_dir():
        for sub in ("imagesTr", "imagesTs"):
            p = images_root / sub
            if p.is_dir():
                for f in sorted(p.glob("*.nii")) + sorted(p.glob("*.nii.gz")):
                    yield f
    else:
        for f in sorted(images_root.glob("*.nii")) + sorted(images_root.glob("*.nii.gz")):
            yield f

def case_id_from_filename(fname: str) -> str:
    # handle case_0000.nii(.gz) or case.nii(.gz)
    base = Path(fname).name
    # remove extensions .nii or .nii.gz
    if base.endswith(".nii.gz"):
        stem = base[:-7]
    elif base.endswith(".nii"):
        stem = base[:-4]
    else:
        stem = Path(base).stem
    # if endswith _0000 or _0001 etc, strip trailing _\d{4}
    if stem.endswith("_0000") or stem.endswith("_0001") or stem.endswith("_0002"):
        return stem.rsplit("_", 1)[0]
    # also handle single-slice naming case_0 etc
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem

def approx_equal(a: Tuple[float,float,float], b: Tuple[float,float,float], tol: float = 1e-6) -> bool:
    return all(abs(x-y) <= tol for x,y in zip(a,b))

def update_image_spacing(img_path: Path, target_spacing: Tuple[float,float,float]) -> Optional[str]:
    img = nib.load(str(img_path))
    hdr = img.header.copy()
    old_zooms = tuple(hdr.get_zooms()[:3])
    if approx_equal(old_zooms, target_spacing):
        return f"SKIP {img_path.name}: spacing already {old_zooms}"
    # prepare new affine: scale columns by ratio target/old
    old_affine = img.affine.copy()
    new_affine = old_affine.copy()
    # protect against zero old zooms
    if any(z == 0 for z in old_zooms):
        return f"ERROR {img_path.name}: old zoom contains zero {old_zooms}"
    ratios = [target_spacing[i] / old_zooms[i] for i in range(3)]
    # Multiply the first three columns (voxel axes) by ratios
    for i in range(3):
        new_affine[:3, i] = new_affine[:3, i] * ratios[i]
    # Update header pixdim via set_zooms
    hdr.set_zooms((*target_spacing,) + tuple(hdr.get_zooms()[3:]))
    # save with new affine and header (keep original data array)
    data = np.asarray(img.dataobj)
    new_img = nib.Nifti1Image(data, new_affine, hdr)
    nib.save(new_img, str(img_path))
    return f"UPDATED {img_path.name}: {old_zooms} -> {target_spacing}"

def main():
    ap = argparse.ArgumentParser(description="Apply spacing from spacing.json into image headers (no resampling).")
    ap.add_argument("--images-root", type=Path, default=Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/test"),
                    help="Path to dataset root containing imagesTr/test, or to a single images folder.")
    ap.add_argument("--spacing", type=Path, default="/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_Fposev2/spacing.json", help="Path to spacing.json (nnUNet_raw dataset).")

    args = ap.parse_args()

    spacing_map = load_spacing_json(args.spacing)
    results = []

    for img_path in iter_image_files(args.images_root):
        cid = case_id_from_filename(img_path.name)
        if cid not in spacing_map:
            results.append(f"MISSING_SPACING {img_path.name}: case id '{cid}' not in spacing.json")
            continue

        if args.skip_if_label_exists:
            label = find_label_for_case(img_path, cid)
            if label is not None:
                results.append(f"SKIP_LABEL {img_path.name}: label exists at {label}")
                continue

        try:
            msg = update_image_spacing(img_path, spacing_map[cid], backup=args.backup, dry_run=args.dry_run)
            results.append(msg)
        except Exception as e:
            results.append(f"ERROR {img_path.name}: {e}")

    # print summary
    updated = [r for r in results if r and (r.startswith("UPDATED") or r.startswith("DRY"))]
    skipped = [r for r in results if r and r.startswith("SKIP")]
    missing = [r for r in results if r and r.startswith("MISSING_SPACING")]
    errors = [r for r in results if r and r.startswith("ERROR")]
    print("Summary:")
    print(f"  processed: {len(results)}")
    print(f"  updated/dry: {len(updated)}")
    print(f"  skipped (already correct): {len(skipped)}")
    print(f"  missing spacing: {len(missing)}")
    print(f"  errors: {len(errors)}")
    # print details for missing/errors
    if missing:
        print("\nMissing spacing entries:")
        for m in missing:
            print("  " + m)
    if errors:
        print("\nErrors:")
        for e in errors:
            print("  " + e)

if __name__ == '__main__':
    main()