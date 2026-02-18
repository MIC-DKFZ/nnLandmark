#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from collections import Counter
from typing import Optional, Tuple, Dict, List, Sequence, Any

# optional dependency: pynrrd (pip install pynrrd)
try:
    import nrrd  # type: ignore
except Exception:
    nrrd = None


def _parse_vector(v: Any) -> Optional[np.ndarray]:
    """Convert various space-directions representations into a numpy vector or None."""
    if v is None:
        return None
    try:
        return np.array(v, dtype=float)
    except Exception:
        try:
            s = str(v).strip('() ')
            parts = [float(p) for p in s.split(',')]
            return np.array(parts, dtype=float)
        except Exception:
            return None


def parse_fcsv(fcsv_path: Path) -> List[Dict]:
    """
    Parse an .fcsv and return list of entries:
      { 'id': str, 'label': str, 'world': (x,y,z) }
    Detect column ordering from '# columns = ...' header if present.
    """
    entries: List[Dict] = []
    col_order: Optional[List[str]] = None
    with fcsv_path.open('r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                if 'columns' in line:
                    try:
                        _, _, cols = line.partition('columns')
                        cols = cols.split('=')[-1].strip()
                        col_order = [c.strip() for c in cols.split(',') if c.strip()]
                    except Exception:
                        col_order = None
                continue
            if not line.strip():
                continue
            fields = line.split(',')
            if col_order:
                idx = {name: i for i, name in enumerate(col_order)}
                try:
                    x = float(fields[idx.get('x', 1)])
                    y = float(fields[idx.get('y', 2)])
                    z = float(fields[idx.get('z', 3)])
                except Exception:
                    try:
                        x, y, z = float(fields[1]), float(fields[2]), float(fields[3])
                    except Exception:
                        continue
                label = ""
                if 'label' in idx and len(fields) > idx['label']:
                    label = fields[idx['label']].strip()
                else:
                    label = fields[-3].strip() if len(fields) >= 3 else ""
                uid = fields[0].strip() if fields else ""
            else:
                try:
                    x, y, z = float(fields[1]), float(fields[2]), float(fields[3])
                except Exception:
                    continue
                label = fields[11].strip() if len(fields) > 11 else ""
                uid = fields[0].strip() if fields else ""
            entries.append({"id": uid, "label": label, "world": (x, y, z)})
    return entries


def get_nrrd_info(nrrd_file: Path) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[float, ...]], Dict]:
    """
    Return (shape, spacing, header) for a nrrd file.
    spacing derived from header['space directions'] (vector norms) or header['spacings'] if present.
    Returns (None, None, {}) if unable to read.
    """
    if nrrd is None:
        raise RuntimeError("pynrrd (nrrd) is not installed. Install with: pip install pynrrd")
    try:
        data, header = nrrd.read(str(nrrd_file))
        shape = tuple(data.shape)
        spacing = None
        sd = header.get('space directions') or header.get('space_directions') or header.get('space-directions')
        if sd:
            spacings = []
            for v in sd:
                vec = _parse_vector(v)
                if vec is None:
                    spacings.append(None)
                else:
                    spacings.append(float(np.linalg.norm(vec)))
            spacing = tuple(float(s) if s is not None else 1.0 for s in spacings)
        else:
            sp = header.get('spacings') or header.get('sizes') or header.get('space origin')
            if sp and isinstance(sp, (list, tuple)) and len(sp) >= len(shape):
                spacing = tuple(float(x) for x in sp[:len(shape)])
        return shape, spacing, header
    except Exception:
        return None, None, {}


def world_to_voxel(world: Sequence[float], header: Dict) -> Optional[Tuple[float, float, float]]:
    """
    Convert world coordinate to voxel/index coordinate using nrrd header fields:
      world = origin + [space_dir_0, space_dir_1, ...] @ index_vector
    Solve index = inv(space_dir_matrix) @ (world - origin)
    Returns None if conversion not possible.
    """
    if header is None:
        return None
    origin = header.get('space origin') or header.get('space_origin') or header.get('space-origin') or header.get('space origin (mm)')
    if origin is None:
        origin = header.get('space origin (mm)') or header.get('origin')
    if origin is None:
        return None
    origin_vec = _parse_vector(origin)
    if origin_vec is None:
        try:
            origin_vec = np.array([float(x) for x in origin])
        except Exception:
            return None
    sd = header.get('space directions') or header.get('space_directions') or header.get('space-directions')
    if not sd:
        return None
    mat_cols = []
    for v in sd:
        vec = _parse_vector(v)
        if vec is None:
            return None
        mat_cols.append(vec)
    mat = np.column_stack(mat_cols)
    try:
        inv = np.linalg.inv(mat)
    except Exception:
        return None
    delta = np.array(world, dtype=float) - origin_vec
    idx = inv.dot(delta)
    return float(idx[0]), float(idx[1]), float(idx[2])


def scan_cases(root_path: str, verbose: bool = True, spacing_tol: float = 1e-3):
    root = Path(root_path)
    if not root.exists():
        raise SystemExit(f"Root path {root} does not exist")

    per_case_fc_count = Counter()
    per_case_landmark_counts = Counter()
    per_fcsv_landmark_counter = Counter()
    cases_with_multifcsv = []
    nrrd_shape_counter = Counter()
    nrrd_spacing_counter = Counter()
    nrrd_shape_examples = {}
    nrrd_spacing_examples = {}
    missing_img_cases = []
    bad_nrrd_cases = []

    label_counter = Counter()
    per_label_examples = {}
    per_case_details = {}

    case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for case_dir in case_dirs:
        print(case_dir.name)
        nrrd_files = sorted(case_dir.glob("*.nrrd")) + sorted(case_dir.glob("*.nhdr"))
        fcsv_files = sorted(case_dir.glob("*.fcsv"))

        img_shape = None
        img_spacing = None
        header = {}

        if not nrrd_files:
            missing_img_cases.append(case_dir.name)
        else:
            img_path = nrrd_files[0]
            try:
                img_shape, img_spacing, header = get_nrrd_info(img_path)
                if img_shape:
                    nrrd_shape_counter[img_shape] += 1
                    nrrd_shape_examples.setdefault(img_shape, img_path.name)
                if img_spacing:
                    rounded = tuple(int(round(s / spacing_tol)) * spacing_tol for s in img_spacing)
                    nrrd_spacing_counter[rounded] += 1
                    nrrd_spacing_examples.setdefault(rounded, img_path.name)
            except RuntimeError:
                raise
            except Exception:
                bad_nrrd_cases.append(case_dir.name)
                img_shape = None
                img_spacing = None
                header = {}

        per_case_fc_count[len(fcsv_files)] += 1
        if len(fcsv_files) > 1:
            cases_with_multifcsv.append((case_dir.name, [f.name for f in fcsv_files]))

        total_landmarks_in_case = 0
        case_entries: List[Dict] = []
        for fcsv in fcsv_files:
            entries = parse_fcsv(fcsv)
            per_fcsv_landmark_counter[len(entries)] += 1
            total_landmarks_in_case += len(entries)
            for e in entries:
                label = e.get('label', '') or ''
                label_counter[label] += 1
                per_label_examples.setdefault(label, []).append(fcsv.name)
                world = e.get('world')
                voxel = None
                if world is not None and header:
                    voxel = world_to_voxel(world, header)
                case_entries.append({"id": e.get('id', ''), "label": label, "world": world, "voxel": voxel, "fcsv": fcsv.name})
        per_case_landmark_counts[total_landmarks_in_case] += 1

        per_case_details[case_dir.name] = {
            "nrrd": nrrd_files[0].name if nrrd_files else None,
            "shape": img_shape,
            "spacing": img_spacing,
            "n_fcsv": len(fcsv_files),
            "n_landmarks": total_landmarks_in_case,
            "entries": case_entries,
        }

        if verbose and (len(fcsv_files) == 0):
            print(f"[WARN] No .fcsv annotation found in {case_dir.name}")

    print("\n=== Dataset scan summary ===")
    print(f"Total cases scanned: {len(case_dirs)}")

    print("\nNumber of .fcsv files per case (histogram):")
    for k, v in sorted(per_case_fc_count.items()):
        print(f"  {k} .fcsv files: {v} cases")
    if cases_with_multifcsv:
        print(f"\nCases with multiple .fcsv files ({len(cases_with_multifcsv)}):")
        for case, files in cases_with_multifcsv:
            print(f"  {case}: {len(files)} files -> {files}")

    print("\nTotal landmarks per case (histogram):")
    for n, cnt in sorted(per_case_landmark_counts.items()):
        print(f"  {n} landmarks: {cnt} cases")

    print("\nLandmarks per single .fcsv (histogram):")
    for n, cnt in sorted(per_fcsv_landmark_counter.items()):
        print(f"  {n} landmarks in .fcsv: {cnt} files")

    print("\nImage shapes (nrrd) histogram:")
    for s, cnt in sorted(nrrd_shape_counter.items()):
        print(f"  {s}: {cnt} cases (example: {nrrd_shape_examples.get(s)})")

    print("\nImage spacings (rounded) histogram:")
    for sp, cnt in sorted(nrrd_spacing_counter.items()):
        print(f"  {sp}: {cnt} cases (example: {nrrd_spacing_examples.get(sp)})")

    print("\nAnnotation label histogram (top 30):")
    for label, cnt in label_counter.most_common(30):
        examples = per_label_examples.get(label, [])[:3]
        print(f"  '{label}': {cnt} occurrences (examples: {examples})")

    if missing_img_cases:
        print(f"\nCases with missing image ({len(missing_img_cases)}): {missing_img_cases[:10]}{'...' if len(missing_img_cases)>10 else ''}")
    if bad_nrrd_cases:
        print(f"\nCases with unreadable nrrd ({len(bad_nrrd_cases)}): {bad_nrrd_cases[:10]}{'...' if len(bad_nrrd_cases)>10 else ''}")

    out_summary = root / "dataset_scan_summary.json"
    try:
        import json
        with out_summary.open("w") as f:
            json.dump({
                "n_cases": len(case_dirs),
                "per_case_details": per_case_details,
                "label_counts": dict(label_counter),
                "shape_histogram": {str(k): v for k, v in nrrd_shape_counter.items()},
                "spacing_histogram": {str(k): v for k, v in nrrd_spacing_counter.items()},
            }, f, indent=2)
        if verbose:
            print(f"\nWrote per-case summary to {out_summary}")
    except Exception:
        pass


if __name__ == "__main__":
    root = "/path/to/2024_Ertl_nnLandmark/data/PDDCA/PDDCA-1.4.1/"
    scan_cases(root, verbose=True)