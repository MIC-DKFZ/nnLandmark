#!/usr/bin/env python3
"""
Simplified: only NRRD inputs.

- Parses .fcsv (Slicer markups) with world (mm) coords.
- Reads .nrrd/.nhdr images only.
- Converts world->voxel using NRRD header (converts Slicer RAS -> NRRD LPS when header indicates LPS).
- Mirrors voxel coordinates in X, Y and XY (keeps mm originals).
- Writes JSON outputs and spacing.json / name_to_label.json.
"""
from pathlib import Path
from collections import OrderedDict
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Sequence, Callable

import numpy as np

try:
    import nrrd  # type: ignore
except Exception:
    nrrd = None

COLS_RE = re.compile(r"columns\s*=\s*(.*)")


def parse_fcsv(fcsv_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    col_order: Optional[List[str]] = None
    with fcsv_path.open("r") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("#"):
                if "columns" in ln:
                    m = COLS_RE.search(ln)
                    if m:
                        col_order = [c.strip() for c in m.group(1).split(",")]
                continue
            if not ln.strip():
                continue
            parts = ln.split(",")
            try:
                if col_order:
                    idx = {name: i for i, name in enumerate(col_order)}
                    x = float(parts[idx.get("x", 1)])
                    y = float(parts[idx.get("y", 2)])
                    z = float(parts[idx.get("z", 3)])
                    label = parts[idx["label"]].strip() if "label" in idx and len(parts) > idx["label"] else ""
                    uid = parts[idx.get("id", 0)].strip() if "id" in idx else parts[0].strip()
                else:
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                    label = parts[11].strip() if len(parts) > 11 else (parts[-3].strip() if len(parts) >= 3 else "")
                    uid = parts[0].strip()
                entries.append({"id": uid, "label": label, "world": (x, y, z)})
            except Exception:
                continue
    return entries


def _extract_spacing_from_nrrd_header(header: Dict) -> Optional[Tuple[float, float, float]]:
    if not header:
        return None
    sd = header.get("space directions")
    if sd is None or (hasattr(sd, "__len__") and len(sd) == 0):
        print("No 'space directions' in NRRD header !!!")
    spacings: List[float] = []
    for v in sd:
        if v is None:
            spacings.append(1.0)
            continue
        try:
            arr = np.array(v, dtype=float)
            spacings.append(float(np.linalg.norm(arr)))
        except Exception:
            try:
                s = str(v).strip("() ")
                parts = [float(p) for p in s.split(",")]
                spacings.append(float(np.linalg.norm(np.array(parts, dtype=float))))
            except Exception:
                spacings.append(1.0)
    if len(spacings) >= 3:
        return (float(spacings[0]), float(spacings[1]), float(spacings[2]))
    return tuple(float(s) for s in spacings)


def _get_origin_from_nrrd_header(header: Dict) -> Optional[np.ndarray]:
    if not header:
        return None
    origin = header.get("space origin")
    if origin is None:
        return None
    try:
        return np.array(origin, dtype=float)
    except Exception:
        try:
            s = str(origin).strip("() ")
            return np.array([float(p) for p in s.split(",")], dtype=float)
        except Exception:
            return None


def world_to_voxel_nrrd(world: Sequence[float], header: Dict) -> Optional[Tuple[float, float, float]]:
    origin = _get_origin_from_nrrd_header(header)
    sd = header.get("space directions")
    if origin is None or sd is None or (hasattr(sd, "__len__") and len(sd) == 0):
        print("No 'space directions' or origin in NRRD header for world to voxel!!!")
        return None
    cols = []
    for v in sd:
        try:
            cols.append(np.array(v, dtype=float))
        except Exception:
            try:
                s = str(v).strip("() ")
                cols.append(np.array([float(p) for p in s.split(",")], dtype=float))
            except Exception:
                return None
    try:
        mat = np.column_stack(cols)
        inv = np.linalg.inv(mat)
    except Exception:
        return None
    delta = np.array(world, dtype=float) - origin
    idx = inv.dot(delta)
    return float(idx[0]), float(idx[1]), float(idx[2])


def _approx_world_to_voxel_by_spacing(world: Sequence[float], spacing: Sequence[float], origin: Optional[Sequence[float]] = None) -> Tuple[float, float, float]:
    w = np.array(world, dtype=float)
    sp = np.array(spacing, dtype=float)
    org = np.array(origin, dtype=float) if origin is not None else np.zeros(3, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = (w - org) / sp
    return float(idx[0]), float(idx[1]), float(idx[2])


def _to_pylist(coord: Sequence[float]) -> List[float]:
    return [float(coord[0]), float(coord[1]), float(coord[2])]


def main():
    if nrrd is None:
        raise SystemExit("pynrrd package is required for this simplified script (only NRRD supported).")

    root = Path("/path/to/2024_Ertl_nnLandmark/data/PDDCA/PDDCA-1.4.1")
    if len(__import__("sys").argv) > 1:
        root = Path(__import__("sys").argv[1]).expanduser()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    all_labels_voxel: Dict[str, Dict[str, List[float]]] = {}
    all_labels_mm: Dict[str, Dict[str, List[float]]] = {}
    #all_labels_voxel_mx: Dict[str, Dict[str, List[float]]] = {}
    #all_labels_voxel_my: Dict[str, Dict[str, List[float]]] = {}
    #all_labels_voxel_mxy: Dict[str, Dict[str, List[float]]] = {}

    spacing_map: Dict[str, Dict[str, Optional[List[float]]]] = {}
    observed_label_names: List[str] = []

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        print(f"Processing case: {case_dir.name}")
        fcsv_files = sorted(case_dir.glob("*.fcsv"))
        if not fcsv_files:
            continue

        nrrd_files = sorted(case_dir.glob("*.nrrd")) + sorted(case_dir.glob("*.nhdr"))
        if not nrrd_files:
            continue

        img_header = None
        img_spacing = None
        img_shape = None
        world_transform: Callable[[Sequence[float]], Sequence[float]] = lambda w: (float(w[0]), float(w[1]), float(w[2]))

        try:
            data, header = nrrd.read(str(nrrd_files[0]))
            img_header = header
            img_shape = data.shape
            img_spacing = _extract_spacing_from_nrrd_header(header)
            space = ""
            try:
                space = str(header.get("space", "")).lower()
            except Exception:
                space = ""
            # header referencing 'left' or 'lps' means NRRD is LPS; Slicer uses RAS -> convert RAS->LPS
            if "left" in space or "lps" in space:
                world_transform = lambda w: (-float(w[0]), -float(w[1]), float(w[2]))
            else:
                world_transform = lambda w: (float(w[0]), float(w[1]), float(w[2]))
        except Exception:
            continue

        case_voxel = OrderedDict()
        case_mm = OrderedDict()
        #case_voxel_mx = OrderedDict()
        #case_voxel_my = OrderedDict()
        #case_voxel_mxy = OrderedDict()

        for fcsv in fcsv_files:
            entries = parse_fcsv(fcsv)
            for e in entries:
                raw_label = (e.get("label") or "").strip()
                if not raw_label:
                    raw = e.get("id", "")
                    m = re.match(r".*?(\d+)$", raw)
                    raw_label = f"pt_{m.group(1)}" if m else raw or "unknown"
                if raw_label not in observed_label_names:
                    observed_label_names.append(raw_label)

                world = e.get("world")
                if world is not None:
                    case_mm[raw_label] = _to_pylist(world)

                voxel = None
                if world is not None:
                    world_for_image = world_transform(world)
                    try:
                        voxel = world_to_voxel_nrrd(world_for_image, img_header)
                    except Exception:
                        voxel = None

                if voxel is None and img_spacing is not None and world is not None:
                    origin = _get_origin_from_nrrd_header(img_header)
                    try:
                        world_for_image = world_transform(world)
                        voxel = _approx_world_to_voxel_by_spacing(world_for_image, img_spacing, origin)
                    except Exception:
                        voxel = None

                if voxel is not None:
                    v = np.array(voxel, dtype=float)
                    case_voxel[raw_label] = _to_pylist(v)

                    if img_shape is not None and len(img_shape) >= 3:
                        vmax = np.array([img_shape[0] - 1, img_shape[1] - 1, img_shape[2] - 1], dtype=float)
                        vx = v.copy(); vx[0] = vmax[0] - v[0]
                        vy = v.copy(); vy[1] = vmax[1] - v[1]
                        vxy = v.copy(); vxy[0] = vmax[0] - vxy[0]; vxy[1] = vmax[1] - vxy[1]
                        #case_voxel_mx[raw_label] = _to_pylist(vx)
                        #case_voxel_my[raw_label] = _to_pylist(vy)
                        #case_voxel_mxy[raw_label] = _to_pylist(vxy)

        if case_voxel:
            all_labels_voxel[case_dir.name] = case_voxel
        if case_mm:
            all_labels_mm[case_dir.name] = case_mm
        #if case_voxel_mx:
        #    all_labels_voxel_mx[case_dir.name] = case_voxel_mx
        #if case_voxel_my:
        #    all_labels_voxel_my[case_dir.name] = case_voxel_my
        #if case_voxel_mxy:
        #    all_labels_voxel_mxy[case_dir.name] = case_voxel_mxy

        spacing_map[case_dir.name] = {
            "image_spacing": list(img_spacing) if img_spacing else None,
            "annotation_spacing": None,
            "image_header_space": (img_header.get("space") if img_header else None),
        }

    name_to_label: Dict[str, int] = {name: i for i, name in enumerate(sorted(observed_label_names), start=1)}

    out_root = Path("/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA")
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "all_landmarks_voxel.json").write_text(json.dumps(all_labels_voxel, indent=2))
    (out_root / "all_landmarks_mm.json").write_text(json.dumps(all_labels_mm, indent=2))
    #(out_root / "all_landmarks_voxel_mirror_x.json").write_text(json.dumps(all_labels_voxel_mx, indent=2))
    #(out_root / "all_landmarks_voxel_mirror_y.json").write_text(json.dumps(all_labels_voxel_my, indent=2))
    #(out_root / "all_landmarks_voxel_mirror_xy.json").write_text(json.dumps(all_labels_voxel_mxy, indent=2))
    (out_root / "spacing.json").write_text(json.dumps(spacing_map, indent=2))
    (out_root / "name_to_label.json").write_text(json.dumps(name_to_label, indent=2))

    print("Wrote JSON outputs to", out_root)


if __name__ == "__main__":
    main()