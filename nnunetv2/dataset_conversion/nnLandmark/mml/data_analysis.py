'''dataset_stats.py

Extended utility to gather dataset statistics *and* show example cases for key spacings.

It now additionally prints **one example filename for spacings that are ~ (0.5, 0.5, 0.5) and (0.5, 0.5, 1.0)**, based on the same rounding tolerance you choose (`--tolerance`).

```bash
python dataset_stats.py                      # scans train, val, test
python dataset_stats.py --split val          # only val
python dataset_stats.py --tolerance 0.05     # broader spacing grouping (mm)
```
'''

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
from collections import Counter

import numpy as np

# Third‑party: pynrrd (pip install pynrrd)
try:
    import nrrd  # type: ignore
except ImportError:
    sys.exit("pynrrd is required (pip install pynrrd)")

###############################################################################
# Helpers
###############################################################################

def round_spacing(spacing: Iterable[float], tol: float) -> Tuple[float, float, float]:
    """Round each element to the nearest multiple of *tol* to reduce float fuzz."""
    return tuple(round(v / tol) * tol for v in spacing)  # type: ignore


def spacing_from_header(header: Dict) -> Tuple[float, float, float]:
    """Extract voxel spacing from a NRRD header."""
    if "space directions" in header and header["space directions"] is not None:
        dirs = header["space directions"]
        return tuple(float(np.linalg.norm(vec)) if vec is not None else 1.0 for vec in dirs)  # type: ignore
    if "spacing" in header:
        return tuple(float(x) for x in header["spacing"])
    raise ValueError("No spacing info in NRRD header")


def classify_landmarks(label: np.ndarray) -> int:
    """Return number of valid landmarks (non‑missing rows)."""
    def is_missing(row: np.ndarray) -> bool:
        return (np.isnan(row).all() or np.allclose(row, 0) or np.allclose(row, -1))

    return int(sum(0 if is_missing(r) else 1 for r in label))


def within_tol(a: Tuple[float, ...], b: Tuple[float, ...], tol: float) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))

###############################################################################
# Scanning logic
###############################################################################

def scan_split(split_dir: Path, tol: float,
               nrrd_counter: Counter,
               npy_counter: Counter,
               landmarks_counter: Counter,
               nrrd_examples: Dict[Tuple[float, float, float], str],
               npy_examples: Dict[Tuple[float, float, float], str],) -> None:
    """Accumulate stats for one split, updating given Counters and example maps."""
    for nrrd_path in sorted(split_dir.glob("*_volume.nrrd")):
        stem = nrrd_path.stem.replace("_volume", "")
        spacing_npy = split_dir / f"{stem}_spacing.npy"
        label_npy = split_dir / f"{stem}_label.npy"

        # --- NRRD spacing ---
        # try:
        #     _, hdr = nrrd.read(nrrd_path, index_order="C")
        #     sp = round_spacing(spacing_from_header(hdr), tol)
        #     nrrd_counter[sp] += 1
        #     nrrd_examples.setdefault(sp, nrrd_path.name)
        # except Exception as e:
        #     print(f"[WARN] Could not read spacing from {nrrd_path.name}: {e}")

        # --- spacing.npy ---
        if spacing_npy.exists():
            try:
                sp_npy = round_spacing(np.load(spacing_npy), tol)  # type: ignore
                npy_counter[sp_npy] += 1
                npy_examples.setdefault(sp_npy, spacing_npy.name)
            except Exception as e:
                print(f"[WARN] Could not load {spacing_npy.name}: {e}")

        # --- labels ---
        if label_npy.exists():
            try:
                lbl = np.load(label_npy)
                if lbl.shape != (14, 3):
                    raise ValueError(f"Expected (14,3), got {lbl.shape}")
                landmarks_counter[classify_landmarks(lbl)] += 1
            except Exception as e:
                print(f"[WARN] Could not load {label_npy.name}: {e}")

###############################################################################
# Reporting helpers
###############################################################################

def print_counter(title: str, counter: Counter):
    print(f"\n=== {title} ===")
    if not counter:
        print("No data available.")
        return
    for k, v in counter.most_common():
        print(f"{k} -> {v} cases")
    print(f"Total unique: {len(counter)} | Total cases: {sum(counter.values())}")


def print_examples(title: str, target_spacings: List[Tuple[float, float, float]],
                   examples: Dict[Tuple[float, float, float], str], tol: float):
    print(f"\n--- Example files for {title} ---")
    for tgt in target_spacings:
        found = next((sp for sp in examples if within_tol(sp, tgt, tol)), None)
        if found:
            print(f"≈ {tgt} -> {examples[found]}")
        else:
            print(f"≈ {tgt} -> (not found)")

def check_landmarks_integrity(split_dir: Path):
    """
    Check all label files in split_dir:
      - Are all (14, 3)?
      - Does every row have 3 present (not-missing) coordinates?
    Print problems and summary.
    """
    print(f"\n[LANDMARK INTEGRITY CHECK] {split_dir.name}")
    files_checked = 0
    shape_problems = []
    missing_lm_problems = []
    landmarks_per_case = Counter()

    for label_path in sorted(split_dir.glob("*_label.npy")):
        files_checked += 1
        arr = np.load(label_path)
        if arr.shape != (14, 3):
            shape_problems.append((label_path.name, arr.shape))
            continue
        valid_count = 0
        for idx, row in enumerate(arr):
            if (
                np.isnan(row).any()
                or np.allclose(row, 0)
                or np.allclose(row, -1)
                or (row < 0).any()
            ):
                missing_lm_problems.append((label_path.name, idx, row))
            else:
                valid_count += 1
        landmarks_per_case[valid_count] += 1

    if not shape_problems and not missing_lm_problems:
        print(f"All {files_checked} label files are OK (shape (14,3), no missing landmark coordinates)!")
    else:
        print(f"\nChecked {files_checked} label files: {len(shape_problems)} shape errors, {len(missing_lm_problems)} missing/invalid landmark rows.")

    if landmarks_per_case:
        print("\nLandmark annotation count histogram:")
        for n, cnt in sorted(landmarks_per_case.items()):
            print(f"  {n} landmarks: {cnt} cases")



###############################################################################
# Main
###############################################################################

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute spacing/landmark stats for mmld_dataset and print example cases")
    parser.add_argument("--base", default="~/dev/Project_SoftDiceLoss/data/mmld_dataset", help="Base dataset directory")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Single split to scan")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Rounding/grouping tolerance (mm)")
    args = parser.parse_args(argv)

    base_dir = Path(os.path.expanduser(args.base))
    if not base_dir.exists():
        sys.exit(f"Base directory not found: {base_dir}")

    splits = [args.split] if args.split else ["train", "val", "test"]

    # Aggregated stats
    nrrd_counter: Counter = Counter()
    npy_counter: Counter = Counter()
    landmarks_counter: Counter = Counter()
    nrrd_examples: Dict[Tuple[float, float, float], str] = {}
    npy_examples: Dict[Tuple[float, float, float], str] = {}

    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"[INFO] Split directory missing: {split_dir}")
            continue
        print(f"\n>>> Scanning split: {split}")
        scan_split(split_dir, args.tolerance, nrrd_counter, npy_counter, landmarks_counter, nrrd_examples, npy_examples)
        check_landmarks_integrity(split_dir)

    # Print aggregated stats
    print_counter("NRRD voxel spacings", nrrd_counter)
    print_counter("Spacing.npy values", npy_counter)
    print_counter("Landmarks per case", landmarks_counter)

    target_spacings = [(0.51, 0.51, 0.5), (0.5, 0.5, 1.0)]
    print_examples("NRRD spacings", target_spacings, nrrd_examples, args.tolerance)
    print_examples("spacing.npy", target_spacings, npy_examples, args.tolerance)


if __name__ == "__main__":
    main()

