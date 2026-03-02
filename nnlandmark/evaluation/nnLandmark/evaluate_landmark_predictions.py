import os, argparse
import numpy as np
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, subfiles

from nnlandmark.dataset_conversion.Dataset119_ToothFairy2_All import load_json
from nnlandmark.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

def load_spacing_map(spacing_json_path: str):
    """
    Returns dict: {case_id: [sx, sy, sz]}.
    Accepts:
      {case: [..]}
      {case: {"annotation_spacing":[..], ...}}  # uses annotation_spacing (preferred)
      {case: {"image_spacing":[..], ...}}        # falls back to image_spacing
    """
    raw = load_json(spacing_json_path)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            if "annotation_spacing" in v and v["annotation_spacing"] is not None:
                if isinstance(v['annotation_spacing'], list):
                    out[k] = tuple(v['annotation_spacing'])
                else:
                    out[k] = tuple([v['annotation_spacing'], v['annotation_spacing'], v['annotation_spacing']])
            elif "image_spacing" in v and v["image_spacing"] is not None:
                out[k] = tuple(v['image_spacing'])
            else:
                raise ValueError(f"Unrecognized dict format for case '{k}': {v}")
        else:
            raise ValueError(f"Unrecognized spacing format for case '{k}': {v}")
    return out

def aggregate_predictions_voxel(pred_dir: Path, label_to_name: dict):
    pred_dir = Path(pred_dir)  # allow str or Path
    out = {}
    for p in sorted(pred_dir.glob("*.json")):
        # skip summaries, per-case mm sidecars, and global aggregates
        if (p.name.startswith("summary")
            or p.name.endswith("_mm.json")
            or p.name in ("prediction_all_landmark_voxel.json", "prediction_all_landmark_mm.json", "dataset.json", "plans.json",
                           "combined_predictions.json", "predict_from_raw_data_args.json", "measurement.json")):
            continue
        case_id = p.stem
        data = load_json(p)
        case_map = {}
        for label_idx_str, payload in data.items():
            if label_idx_str in ("spacing", "background"):
                continue

            # Resolve label index -> landmark name using provided mapping.
            # label_to_name maps label strings (e.g. "1") -> landmark name (e.g. "chin").
            lm_name = None
            # If the key is an index string that exists in label_to_name, use mapped name
            if label_idx_str in label_to_name:
                lm_name = label_to_name[label_idx_str]
            else:
                # If the key already is a landmark name (one of the mapping values), accept it
                if label_idx_str in set(label_to_name.values()):
                    lm_name = label_idx_str
                else:
                    # try to handle numeric-like keys (e.g. 1 as int)
                    try:
                        lab_key = str(int(float(label_idx_str)))
                        lm_name = label_to_name.get(lab_key, None)
                    except Exception:
                        lm_name = None

            if lm_name is None:
                print(f"WARNING: Could not resolve landmark name for label index '{label_idx_str}' in case '{case_id}'. Skipping this entry.")
                continue

            # payload expected to contain "coordinates"
            coords = payload.get("coordinates", payload) if isinstance(payload, dict) else payload
            case_map[lm_name] = list(coords)
        out[case_id] = case_map
    return out

def evaluate_MRE(folder_with_pred_jsons: str, gt_json: str):
    """
    Reads all single case prediction JSONs in `folder_with_pred_jsons` and the GT JSON, and
    Computes MRE (micro), std (micro), std_macro_class (macro), MRE and std for each class in voxel space.
    Writes: summary_voxel.json
    """
    predicted_jsons = [i for i in subfiles(folder_with_pred_jsons, suffix='.json', join=False)
        if i not in ('summary.json', 'summary_voxel.json', 'summary_mm.json', 'dataset.json', 'plans.json', 'combined_predictions.json', 'measurement.json',
                     'prediction_all_landmark_voxel.json', 'prediction_all_landmark_mm.json', 'predict_from_raw_data_args.json', 'summary_mm_image_spacing.json', 'summary_mm_annotation_spacing.json', 'renamed_landmarks.json')
        and not i.endswith('_mm.json')]

    name_label_dict = load_json(os.path.join(os.path.dirname(gt_json), 'name_to_label.json'))
    all_landmarks = name_label_dict.keys()

    gt = load_json(gt_json)

    predicted_identifiers = [i[:-5] for i in predicted_jsons]
    not_in_gt = [i for i in predicted_identifiers if i not in gt.keys()]
    not_in_pred = [i for i in gt.keys() if i not in predicted_identifiers]
    assert len(not_in_gt) == 0, f'There are identifiers in the prediction that are not in the GT. Cannot run script.\nNot in gt: {not_in_gt}'
    if len(not_in_pred) != 0:
        print(f'WARNING! Not all identifiers from the ground truth are found in the prediction. This can be intentional or not. GT: {len(gt.keys())}, pred: {len(predicted_identifiers)} identifiers')

    errors = {i: list() for i in all_landmarks}
    detailed_results = {}
    micro_errors = []

    for k in gt.keys():
        if k in not_in_pred:
            continue

        gt_here = gt[k]
        pred_here = load_json(join(folder_with_pred_jsons, k + '.json'))
        detailed_results[k] = {}

        for ki_gt in gt_here.keys():
            ki_pred = str(name_label_dict[ki_gt])
            pred_coords = pred_here[ki_pred]['coordinates']
            gt_coords = gt_here[ki_gt]
            dist_voxel = float(np.linalg.norm([p - g for p, g in zip(pred_coords, gt_coords)]))
            detailed_results[k][ki_gt] = float(np.round(dist_voxel, decimals=5))
            errors[ki_gt].append(dist_voxel)
            micro_errors.append(dist_voxel)

    # Per-class
    mre_by_landmark = {k: np.mean(errors[k]) for k in errors.keys()}
    std_by_landmark = {k: np.std(errors[k]) for k in errors.keys()}

    # Macro between classes
    std_macro_class = float(np.std(list(mre_by_landmark.values())))

    # All micro-level errors
    mre_micro = float(np.mean(micro_errors))
    std_micro = float(np.std(micro_errors))

    save_json({
        'MRE_micro': float(np.round(mre_micro, 5)),
        'std_micro': float(np.round(std_micro, 5)),
        'std_macro_class': float(np.round(std_macro_class, 5)),
        'MRE_by_class': {i: float(np.round(mre_by_landmark[i], 5)) for i in mre_by_landmark.keys()},
        'std_by_class': {i: float(np.round(std_by_landmark[i], 5)) for i in std_by_landmark.keys()},
        'detailed_results': detailed_results
    }, join(folder_with_pred_jsons, 'summary_voxel.json'), sort_keys=False)


def evaluate_MRE_mm(folder_with_pred_jsons: str, gt_json: str, spacing_json: str):
    """
    Reads all single case prediction JSONs in `folder_with_pred_jsons` and the GT JSON, and
    Computes MRE (micro), std (micro), std_macro_class (macro), MRE and std for each class,
    and Success Rate Distances (SRD) at 2mm, 3mm, and 4mm.
    Writes: summary_mm.json
    """
    predicted_jsons = [i for i in subfiles(folder_with_pred_jsons, suffix='.json', join=False)
        if i not in ('summary.json', 'summary_voxel.json', 'summary_mm.json', 'dataset.json', 'plans.json', 'combined_predictions.json', 'measurement.json',
                     'prediction_all_landmark_voxel.json', 'prediction_all_landmark_mm.json', 'predict_from_raw_data_args.json', 'summary_mm_image_spacing.json', 'summary_mm_annotation_spacing.json', 'renamed_landmarks.json')
        and not i.endswith('_mm.json')]

    name_label_dict = load_json(os.path.join(os.path.dirname(gt_json), 'name_to_label.json'))
    all_landmarks = name_label_dict.keys()

    gt = load_json(gt_json)
    spacing_by_case = load_spacing_map(spacing_json)

    predicted_identifiers = [i[:-5] for i in predicted_jsons]
    not_in_gt = [i for i in predicted_identifiers if i not in gt.keys()]
    not_in_pred = [i for i in gt.keys() if i not in predicted_identifiers]
    #assert len(not_in_gt) == 0, (f'There are identifiers in the prediction that are not in the GT. '
    #                             f'Cannot run script.\nNot in gt: {not_in_gt}')
    if len(not_in_pred) != 0:
        print(f'WARNING! Not all identifiers from the ground truth are found in the prediction. '
              f'This can be intentional or not. GT: {len(gt.keys())}, pred: {len(predicted_identifiers)} identifiers')

    errors = {i: list() for i in all_landmarks}
    detailed_results = {}
    micro_errors = []

    for k in gt.keys():
        if k in not_in_pred:
            continue
        if k not in spacing_by_case:
            raise KeyError(f"No spacing for case '{k}' in {spacing_json}")
        sx, sy, sz = spacing_by_case[k]

        gt_here = gt[k]
        pred_here = load_json(join(folder_with_pred_jsons, k + '.json'))
        detailed_results[k] = {}
        for ki_gt in gt_here.keys():
            ki_pred = str(name_label_dict[ki_gt])
            pred_coords = pred_here[ki_pred]['coordinates']
            gt_coords = gt_here[ki_gt]
            dx = (pred_coords[0] - gt_coords[0]) * sx
            dy = (pred_coords[1] - gt_coords[1]) * sy
            dz = (pred_coords[2] - gt_coords[2]) * sz
            dist_mm = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            detailed_results[k][ki_gt] = float(np.round(dist_mm, decimals=5))
            errors[ki_gt].append(dist_mm)
            micro_errors.append(dist_mm)

    # Per-class means and stds
    mre_by_landmark = {k: np.mean(errors[k]) for k in errors.keys()}
    std_by_landmark = {k: np.std(errors[k]) for k in errors.keys()}

    std_macro_class = float(np.std(list(mre_by_landmark.values())))
    mre_micro = float(np.mean(micro_errors))
    std_micro = float(np.std(micro_errors))

    # Calculate SRD for 2mm, 3mm, 4mm thresholds
    srd_2mm = float(np.mean(np.array(micro_errors) < 2.0))
    srd_3mm = float(np.mean(np.array(micro_errors) < 3.0))
    srd_4mm = float(np.mean(np.array(micro_errors) < 4.0))

    save_json({
        'MRE_micro': float(np.round(mre_micro, 5)),
        'std_micro': float(np.round(std_micro, 5)),
        'std_macro_class': float(np.round(std_macro_class, 5)),
        'SRD_2mm': float(np.round(srd_2mm, 5)),
        'SRD_3mm': float(np.round(srd_3mm, 5)),
        'SRD_4mm': float(np.round(srd_4mm, 5)),
        'MRE_by_class': {i: float(np.round(mre_by_landmark[i], 5)) for i in mre_by_landmark.keys()},
        'std_by_class': {i: float(np.round(std_by_landmark[i], 5)) for i in std_by_landmark.keys()},
        'detailed_results': detailed_results
    }, join(folder_with_pred_jsons, 'summary_mm.json'), sort_keys=False)

    print(f"MRE_micro: {mre_micro:.5f}, std_micro: {std_micro:.5f}, SRD @ 2mm: {srd_2mm:.5f}, @ 3mm: {srd_3mm:.5f}, @ 4mm: {srd_4mm:.5f}")

def evaluate_MRE_from_aggregated(folder_with_pred_jsons: str, gt_json: str):
    """
    Reads the already aggregated prediction.json in `folder_with_pred_jsons` and the GT JSON, and
    Computes MRE (micro), std (micro), std_macro_class (macro), MRE and std for each class in voxel space.
    Reads predictions from the aggregated `prediction_all_landmark_voxel.json` file.
    Writes: summary_voxel.json
    """
    aggregated_pred_path = join(folder_with_pred_jsons, "prediction_all_landmark_voxel.json")
    if not os.path.exists(aggregated_pred_path):
        raise FileNotFoundError(f"Aggregated prediction file not found: {aggregated_pred_path}")

    # Load aggregated predictions and ground truth
    pred_voxel_by_case = load_json(aggregated_pred_path)
    gt = load_json(gt_json)

    name_label_dict = load_json(os.path.join(os.path.dirname(gt_json), 'name_to_label.json'))
    all_landmarks = name_label_dict.keys()

    not_in_pred = [i for i in gt.keys() if i not in pred_voxel_by_case.keys()]
    if len(not_in_pred) != 0:
        print(f'WARNING! Not all identifiers from the ground truth are found in the prediction. '
              f'This can be intentional or not. GT: {len(gt.keys())}, pred: {len(pred_voxel_by_case.keys())} identifiers')

    errors = {i: list() for i in all_landmarks}
    detailed_results = {}
    micro_errors = []

    for k in gt.keys():
        if k in not_in_pred:
            continue

        gt_here = gt[k]
        pred_here = pred_voxel_by_case.get(k, {})
        detailed_results[k] = {}

        for ki_gt in gt_here.keys():
            pred_coords = pred_here.get(ki_gt, None)
            if pred_coords is None:
                print(f"WARNING: Missing prediction for landmark '{ki_gt}' in case '{k}'")
                continue

            gt_coords = gt_here[ki_gt]
            dist_voxel = float(np.linalg.norm([p - g for p, g in zip(pred_coords, gt_coords)]))
            detailed_results[k][ki_gt] = float(np.round(dist_voxel, decimals=5))
            errors[ki_gt].append(dist_voxel)
            micro_errors.append(dist_voxel)

    # Per-class
    mre_by_landmark = {k: np.mean(errors[k]) for k in errors.keys()}
    std_by_landmark = {k: np.std(errors[k]) for k in errors.keys()}

    # Macro between classes
    std_macro_class = float(np.std(list(mre_by_landmark.values())))

    # All micro-level errors
    mre_micro = float(np.mean(micro_errors))
    std_micro = float(np.std(micro_errors))

    save_json({
        'MRE_micro': float(np.round(mre_micro, 5)),
        'std_micro': float(np.round(std_micro, 5)),
        'std_macro_class': float(np.round(std_macro_class, 5)),
        'MRE_by_class': {i: float(np.round(mre_by_landmark[i], 5)) for i in mre_by_landmark.keys()},
        'std_by_class': {i: float(np.round(std_by_landmark[i], 5)) for i in std_by_landmark.keys()},
        'detailed_results': detailed_results
    }, join(folder_with_pred_jsons, 'summary_voxel.json'), sort_keys=False)


def evaluate_MRE_mm_from_aggregated(folder_with_pred_jsons: str, gt_json: str, spacing_json: str):
    """
    Reads the already aggregated prediction.json in `folder_with_pred_jsons` and the GT JSON, and
    Computes MRE (micro), std (micro), std_macro_class (macro), MRE and std for each class,
    and Success Rate Distances (SRD) at 2mm, 3mm, and 4mm.
    Reads predictions from the aggregated `prediction_all_landmark_voxel.json` file.
    Writes: summary_mm.json
    """
    aggregated_pred_path = join(folder_with_pred_jsons, "prediction_all_landmark_voxel.json")
    if not os.path.exists(aggregated_pred_path):
        raise FileNotFoundError(f"Aggregated prediction file not found: {aggregated_pred_path}")

    # Load aggregated predictions, ground truth, and spacing
    pred_voxel_by_case = load_json(aggregated_pred_path)
    gt = load_json(gt_json)
    spacing_by_case = load_spacing_map(spacing_json)

    name_label_dict = load_json(os.path.join(os.path.dirname(gt_json), 'name_to_label.json'))
    all_landmarks = name_label_dict.keys()

    not_in_pred = [i for i in gt.keys() if i not in pred_voxel_by_case.keys()]
    if len(not_in_pred) != 0:
        print(f'WARNING! Not all identifiers from the ground truth are found in the prediction. '
              f'This can be intentional or not. GT: {len(gt.keys())}, pred: {len(pred_voxel_by_case.keys())} identifiers')

    errors = {i: list() for i in all_landmarks}
    detailed_results = {}
    micro_errors = []

    for k in gt.keys():
        if k in not_in_pred:
            continue
        if k not in spacing_by_case:
            raise KeyError(f"No spacing for case '{k}' in {spacing_json}")
        sx, sy, sz = spacing_by_case[k]

        gt_here = gt[k]
        pred_here = pred_voxel_by_case.get(k, {})
        detailed_results[k] = {}
        for ki_gt in gt_here.keys():
            pred_coords = pred_here.get(ki_gt, None)
            if pred_coords is None:
                print(f"WARNING: Missing prediction for landmark '{ki_gt}' in case '{k}'")
                continue

            gt_coords = gt_here[ki_gt]
            dx = (pred_coords[0] - gt_coords[0]) * sx
            dy = (pred_coords[1] - gt_coords[1]) * sy
            dz = (pred_coords[2] - gt_coords[2]) * sz
            dist_mm = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            detailed_results[k][ki_gt] = float(np.round(dist_mm, decimals=5))
            errors[ki_gt].append(dist_mm)
            micro_errors.append(dist_mm)

    # Per-class means and stds
    mre_by_landmark = {k: np.mean(errors[k]) for k in errors.keys()}
    std_by_landmark = {k: np.std(errors[k]) for k in errors.keys()}

    std_macro_class = float(np.std(list(mre_by_landmark.values())))
    mre_micro = float(np.mean(micro_errors))
    std_micro = float(np.std(micro_errors))

    # Calculate SRD for 2mm, 3mm, 4mm thresholds
    srd_2mm = float(np.mean(np.array(micro_errors) < 2.0))
    srd_3mm = float(np.mean(np.array(micro_errors) < 3.0))
    srd_4mm = float(np.mean(np.array(micro_errors) < 4.0))

    save_json({
        'MRE_micro': float(np.round(mre_micro, 5)),
        'std_micro': float(np.round(std_micro, 5)),
        'std_macro_class': float(np.round(std_macro_class, 5)),
        'SRD_2mm': float(np.round(srd_2mm, 5)),
        'SRD_3mm': float(np.round(srd_3mm, 5)),
        'SRD_4mm': float(np.round(srd_4mm, 5)),
        'MRE_by_class': {i: float(np.round(mre_by_landmark[i], 5)) for i in mre_by_landmark.keys()},
        'std_by_class': {i: float(np.round(std_by_landmark[i], 5)) for i in std_by_landmark.keys()},
        'detailed_results': detailed_results
    }, join(folder_with_pred_jsons, 'summary_mm.json'), sort_keys=False)

    print(f"MRE_micro: {mre_micro:.5f}, std_micro: {std_micro:.5f}, SRD @ 2mm: {srd_2mm:.5f}, @ 3mm: {srd_3mm:.5f}, @ 4mm: {srd_4mm:.5f}")


def evaluate_entry_point():
    parser = argparse.ArgumentParser(description="Evaluate nnLandmark predictions (voxel + mm).")
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Dataset ID, e.g. 732"
    )
    parser.add_argument(
        "-pred",
        type=Path,
        required=True,
        help="Path to validation prediction folder"
    )
    args = parser.parse_args()

    nnLandmark_raw = os.environ.get("nnLM_raw")
    if nnLandmark_raw is None:
        raise EnvironmentError("Environment variable 'nnLM_raw' is not set.")
    dataset_name = maybe_convert_to_dataset_name(args.d)
    dataset_root = Path(nnLandmark_raw) / dataset_name

    # load jsons 
    name_to_label_path = dataset_root / "name_to_label.json"
    spacing_json_path = dataset_root / "spacing.json"

    n2l = load_json(name_to_label_path)
    label_to_name = {str(v): k for k, v in n2l.items()}

    # aggregate voxel prediction
    pred_voxel_by_case = aggregate_predictions_voxel(args.pred, label_to_name)
    save_json(pred_voxel_by_case, args.pred / "prediction_all_landmark_voxel.json")

    # evaluate in voxel space
    evaluate_MRE_from_aggregated(args.pred, str(dataset_root / "all_landmarks_voxel.json"))

    # evaluate in mm
    evaluate_MRE_mm_from_aggregated(
        args.pred,
        str(dataset_root / "all_landmarks_voxel.json"),
        str(spacing_json_path)
    )

if __name__ == "__main__":
    evaluate_entry_point()