#!/usr/bin/env python3
import json, argparse
from pathlib import Path

import numpy as np
import nibabel as nib

# ------------------------------------------------------------------ helpers
def draw_cube(seg, center, *, half: int, label: int):
    x, y, z = (int(round(c)) for c in center)
    seg[max(x-half,0):x+half+1,
        max(y-half,0):y+half+1,
        max(z-half,0):z+half+1] = label

# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",      default="/home/a332l/dev/Project_SoftDiceLoss/data/afids-data/data/datasets/",
                   help="AFIDs datasets root")
    p.add_argument("--landmarks", default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset733_Afids/afids_all_landmarks.json",
                   help="JSON: case → { landmark_<n>: [i,j,k] }")
    p.add_argument("--name2label", default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset733_Afids/afids_name_to_label.json",
                   help="JSON: landmark_<n> → int label  (default: derive on the fly)")
    p.add_argument("--output",    default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset733_Afids/labelsAll/",
                   help="Destination for the segmentation masks")
    p.add_argument("--half", type=int, default=1,
                   help="Half cube size (default: 1 → 3×3×3)")
    args = p.parse_args()

    base = Path(args.base)
    out  = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # 1) load landmarks JSON
    all_lms = json.loads(Path(args.landmarks).read_text())

    # 2) load or derive name→label
    if args.name2label:
        name_to_label = json.loads(Path(args.name2label).read_text())
    else:
        name_to_label = {
            f"landmark_{i}": i
            for i in range(1, 33)
        }

    print(f"Using {len(name_to_label)} labels")

    # 3) for each case_key = "<dataset>_sub-XXXXX"
    for case_key, lms in all_lms.items():
        dataset, rest = case_key.split("_", 1)           # split once
        sub_id = rest                                    # e.g. "sub-103111"
        anat = base / dataset / sub_id / "anat"
        nifs = sorted(anat.glob("*.nii*"))
        if not nifs:
            print(f"[WARN] no image for {case_key}")
            continue

        img = nib.load(str(nifs[0]))
        seg = np.zeros(img.shape, dtype=np.uint8)

        for name, label in name_to_label.items():
            if name in lms:
                draw_cube(seg, lms[name], half=args.half, label=label)
            else:
                print(f"[WARN] {name} missing in {case_key}")

        out_file = out / f"{case_key}.nii.gz"
        nib.save(nib.Nifti1Image(seg, img.affine, img.header),
                 str(out_file))
        print(f"Wrote {out_file.name}")

if __name__ == "__main__":
    main()