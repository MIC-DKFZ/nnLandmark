#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

def draw_cube(seg, center, half=1, label=1):
    x, y, z = (int(round(c)) for c in center)
    xs = slice(max(x-half, 0), x+half+1)
    ys = slice(max(y-half, 0), y+half+1)
    zs = slice(max(z-half, 0), z+half+1)
    seg[xs, ys, zs] = label

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",   default="/home/a332l/dev/Project_SoftDiceLoss/data/landmark_datasets/pose_dataset/volumes/", 
                   help="Pose volumes folder")
    p.add_argument("--landmarks", default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset732_Fetal_pose/pose_dataset_all_landmarks_voxel.json", 
                   help="JSON of landmarks")
    p.add_argument("--name2label", default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset732_Fetal_pose/pose_dataset_name_to_label.json",
                   help="JSON: landmark_<n> → int label  (default: derive on the fly)")
    p.add_argument("--output",    default="/home/a332l/dev/Project_SoftDiceLoss/nnunet_data/nnUNet_raw/Dataset732_Fetal_pose/labelsAll/", 
                   help="where to write segmaps")
    p.add_argument("--half", type=int, default=1,
                   help="Half cube size (default: 1 → 3×3×3)")
    args = p.parse_args()

    base = Path(args.base)
    out  = Path(args.output); out.mkdir(exist_ok=True, parents=True)

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

    for case_key, lms in all_lms.items():
        # find .nii or .nii.gz
        nifs = sorted(base.glob(f"{case_key}.nii")) + sorted(base.glob(f"{case_key}.nii.gz"))
        if not nifs:
            print(f"[WARN] no image for {case_key}")
            continue
        img = nib.load(str(nifs[0]))
        arr = img.get_fdata()
        seg = np.zeros(arr.shape, dtype=np.uint8)

        for name, label in name_to_label.items():
            if name in lms:
                coords = lms[name]
                draw_cube(seg, coords, half=1, label=label)
            else:
                print("WARNING: name not in landmarks!!!")

        seg_img = nib.Nifti1Image(seg, img.affine, img.header)
        out_path = out / f"{case_key}.nii.gz"
        nib.save(seg_img, str(out_path))
        print(f"Wrote {out_path.name}")

if __name__=="__main__":
    main()
