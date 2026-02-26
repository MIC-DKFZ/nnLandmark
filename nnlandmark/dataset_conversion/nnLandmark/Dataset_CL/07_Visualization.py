#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser(
        description="Make 2D multi-class label maps (BMP) from landmark JSON + overlay visualizations."
    )
    p.add_argument(
        "--images_dir",
        type=Path,
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/imagesTr",
        help="Folder containing images (e.g. .../imagesTr with files like CLD2D_<stem>_0000.bmp).",
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
    p.add_argument(
        "--viz_dir",
        type=Path,
        default="/home/a332l/dev/Project_nnLandmark/nnunet_data/nnUNet_raw/Dataset738_CL/Visualized",
        help="Folder for overlay visualizations (PNG).",
    )
    p.add_argument("--prefix", default="", help="Case-ID prefix used when naming files.")
    p.add_argument("--radius_px", type=int, default=10, help="Disc radius in pixels for label map.")
    p.add_argument("--square_half", type=int, default=5, help="Half-size of 2D square for viz (2 => 5x5).")
    p.add_argument("--viz_alpha", type=int, default=210, help="Alpha (0-255) for overlay fill.")
    p.add_argument("--outline_thickness", type=int, default=3, help="Outline stroke width for squares.")
    return p.parse_args()


def find_image_for_case(images_dir: Path, case_id: str, prefix: str) -> Path | None:
    """Try common patterns for original image lookup."""
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
    """In-place filled disc for the label map."""
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
        row = mask[y]
        for x in range(x0, x1 + 1):
            if (x - cx_r) ** 2 + dy2 <= rr:
                row[x] = label


def color_from_label(label: int) -> tuple[int, int, int]:
    """
    Deterministic pseudo-rainbow color from label id.
    Keeps labels visually distinct without a huge palette.
    """
    # simple hash-based palette
    #np.random.seed(label * 9176 + 1337)
    #return tuple(int(x) for x in np.random.randint(32, 255, size=3))
    return (0, 255, 0)



def draw_square(draw: ImageDraw.ImageDraw, x: float, y: float, half: int, fill_rgba: tuple, outline_rgb: tuple | None, width: int = 1):
    """Draw a filled square centered at (x,y) with side = 2*half+1."""
    cx, cy = int(round(x)), int(round(y))
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    draw.rectangle([x0, y0, x1, y1], fill=fill_rgba, outline=outline_rgb, width=width)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

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
            # grayscale-to-RGB for viz
            img_rgb = im.convert("RGB")

        # ---------- 1) label map (uint8 BMP) ----------
        mask = np.zeros((H, W), dtype=np.uint8)
        for name, label in name2label.items():
            if name in lmks:
                x, y = lmks[name]
                if not (np.isfinite(x) and np.isfinite(y)) or x < 0 or y < 0:
                    missing_points += 1
                    continue
                paint_disc(mask, float(x), float(y), args.radius_px, int(label))
        out_mask = args.out_dir / f"{case_id}.bmp"
        Image.fromarray(mask, mode="L").save(out_mask)
        created += 1
        print(f"Wrote {out_mask.name}  (size {W}x{H})")

        # ---------- 2) overlay visualization (PNG) ----------
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for name, label in name2label.items():
            if name in lmks:
                x, y = lmks[name]
                if not (np.isfinite(x) and np.isfinite(y)) or x < 0 or y < 0:
                    continue
                rgb = color_from_label(int(label))
                rgba = (rgb[0], rgb[1], rgb[2], int(np.clip(args.viz_alpha, 0, 255)))
                draw_square(draw, float(x), float(y), args.square_half, rgba, rgb, width=args.outline_thickness)

        viz = Image.alpha_composite(img_rgb.convert("RGBA"), overlay).convert("RGB")
        out_viz = args.viz_dir / f"{case_id}.png"
        viz.save(out_viz)
        print(f"Wrote {out_viz.name}")

    print(f"\n✅ wrote {created} label maps → {args.out_dir}")
    print(f"✅ wrote overlays → {args.viz_dir}")
    if missing_imgs:
        print(f"[WARN] {len(missing_imgs)} cases missing images (first 5): {missing_imgs[:5]}")
    if missing_points:
        print(f"[WARN] skipped {missing_points} invalid landmark points (NaN/negative).")


if __name__ == "__main__":
    main()
