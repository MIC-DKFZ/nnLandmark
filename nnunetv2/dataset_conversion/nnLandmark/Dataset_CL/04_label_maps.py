#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Make 2D multi-class label maps (BMP) from landmark JSON."
    )
    p.add_argument(
        "--images_dir",
        type=Path,
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/imagesTr/",
        help="Folder containing images (e.g. .../imagesAll with files like CLD2D_<stem>_0000.bmp).",
    )
    p.add_argument(
        "--coordinates_json",
        type=Path,
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/landmark_coordinates.json",
        help="Coordinates JSON: {case_id: {name: [x,y], ...}, ...}",
    )
    p.add_argument(
        "--name_to_label_json",
        type=Path,
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/name_to_label.json",
        help="Mapping JSON: {landmark_name: class_id, ...}",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/labelsTr",
        help="Output folder for BMP label maps.",
    )
    p.add_argument("--prefix", default="CLD2D_", help="Case-ID prefix used when naming files.")
    p.add_argument("--radius_px", type=int, default=3, help="Disc radius in pixels.")
    return p.parse_args()


def find_image_for_case(images_dir: Path, case_id: str, prefix: str) -> Path | None:
    """
    Try common patterns:
      1) <case_id>_0000.<ext>
      2) <stem-of-case-id-without-prefix>.<ext>   (if you run on the original images dir)
    """
    stem_no_prefix = case_id[len(prefix):] if case_id.startswith(prefix) else case_id
    patterns = []
    for ext in (".bmp", ".BMP", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        patterns.append(images_dir / f"{case_id}_0000{ext}")
    for ext in (".bmp", ".BMP", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        patterns.append(images_dir / f"{stem_no_prefix}{ext}")

    for p in patterns:
        if p.exists():
            return p
    return None


def paint_disc(mask: np.ndarray, cx: float, cy: float, r: int, label: int):
    """In-place filled disc, clipped to image bounds."""
    H, W = mask.shape
    x0 = max(int(round(cx)) - r, 0)
    x1 = min(int(round(cx)) + r, W - 1)
    y0 = max(int(round(cy)) - r, 0)
    y1 = min(int(round(cy)) + r, H - 1)
    rr = r * r
    cy_r = int(round(cy))
    cx_r = int(round(cx))
    for y in range(y0, y1 + 1):
        dy2 = (y - cy_r) ** 2
        # slice fill per row for speed
        row = mask[y]
        for x in range(x0, x1 + 1):
            if (x - cx_r) ** 2 + dy2 <= rr:
                row[x] = label


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coords = json.loads(args.coordinates_json.read_text())
    name2label = json.loads(args.name_to_label_json.read_text())

    created = 0
    missing_imgs = []
    missing_points = 0

    for case_id, lmks in coords.items():
        img_path = find_image_for_case(args.images_dir, case_id, args.prefix)
        if img_path is None:
            print(f"[WARN] no image found for {case_id}")
            missing_imgs.append(case_id)
            continue

        # determine H×W from the actual image
        with Image.open(img_path) as im:
            W, H = im.size

        # 8-bit mask
        mask = np.zeros((H, W), dtype=np.uint8)

        for name, label in name2label.items():
            if name in lmks:
                x, y = lmks[name]
                # guard NaNs / negatives
                if not (np.isfinite(x) and np.isfinite(y)) or x < 0 or y < 0:
                    missing_points += 1
                    continue
                paint_disc(mask, float(x), float(y), args.radius_px, int(label))
            else:
                # landmark absent in this image (ok)
                pass

        # write as BMP (L mode preserves class ids 0..255)
        out_path = args.out_dir / f"{case_id}.bmp"
        Image.fromarray(mask, mode="L").save(out_path)
        created += 2
        print(f"Wrote {out_path.name}  (size {W}x{H})")

    print(f"\n✅ wrote {created} label maps → {args.out_dir}")
    if missing_imgs:
        print(f"[WARN] {len(missing_imgs)} cases missing images (first 5): {missing_imgs[:5]}")
    if missing_points:
        print(f"[WARN] skipped {missing_points} invalid landmark points (NaN/negative).")


if __name__ == "__main__":
    main()

