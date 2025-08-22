import os, argparse
import numpy as np
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, subfiles

from nnunetv2.dataset_conversion.Dataset119_ToothFairy2_All import load_json

def load_spacing_map(spacing_json_path: str):
    """
    Returns dict: {case_id: [sx, sy, sz]}.
    Accepts:
      {case: [..]}
      {case: {"spacing":[..]}}
      {case: {"image_spacing":[..], ...}}  # uses image_spacing
    """
    raw = load_json(spacing_json_path)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            if "image_spacing" in v and v["image_spacing"] is not None:
                out[k] = [float(x) for x in v["image_spacing"]]
            elif "annotation_spacing" in v and v["annotation_spacing"] is not None:
                out[k] = [float(x) for x in v["annotation_spacing"]]
            else:
                raise ValueError(f"Unrecognized dict format for case '{k}': {v}")
        elif isinstance(v, (list, tuple)) and len(v) == 3:
            out[k] = [float(x) for x in v]
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
            or p.name in ("prediction_all_landmark_voxel.json", "prediction_all_landmark_mm.json")):
            continue
        case_id = p.stem
        data = load_json(p)
        case_map = {}
        for label_idx_str, payload in data.items():
            if label_idx_str == "background":
                continue
            lm_name = label_to_name.get(label_idx_str)
            if lm_name is None:
                continue
            case_map[lm_name] = list(payload["coordinates"])
        out[case_id] = case_map
    return out

def evaluate_MRE(folder_with_pred_jsons: str, gt_json: str):
    """
    IMPORTANT this function only computes the MRE for all landmarks in the GT.
    It DOES NOT evaluate landmark detection, so whether landmarks are predicted that are not in the GT! I will just
    take the coordinate of each landmark, irrespective of its predicted likelihood.
    So this function can only be used for datasets where all landmarks are present in all images!

    If this is not the case, a more sophisticated evaluation scheme is needed where we evaluate MRE and a landmark detection metric

    TODO this script currently only considers pixel distances and does not take into account the voxel spacing!
    """
    # folder_with_pred_jsons = '/home/isensee/drives/checkpoints/nnUNet_results/Dataset737_FPOSE/nnLandmark_trainer__nnUNetResEncUNetLPlans__3d_fullres/crossval_predictions'
    # gt_json = '/home/isensee/drives/E132-Rohdaten/nnUNetv2/Dataset737_FPOSE/all_landmarks_voxel.json'
    predicted_jsons = [i for i in subfiles(folder_with_pred_jsons, suffix='.json', join=False)
                   if i not in ('summary.json', 'summary_voxel.json', 'summary_mm.json', 'dataset.json', 'plans.json', 'combined_predictions.json',
                                    'prediction_all_landmark_voxel.json', 'prediction_all_landmark_mm.json', 'predict_from_raw_data_args.json')
                   and not i.endswith('_mm.json')]
    # we always predict something for all landmarks, so we can infer how many landmarks there are from any model output json
    name_label_dict = load_json(os.path.join(os.path.dirname(gt_json), 'name_to_label.json'))
    all_landmarks = name_label_dict.keys() # [int(i) for i in load_json(join(folder_with_pred_jsons, predicted_jsons[0])).keys()]
    
    gt = load_json(gt_json)
    
    predicted_identifiers = [i[:-5] for i in predicted_jsons]
    not_in_gt = [i for i in predicted_identifiers if i not in gt.keys()]
    not_in_pred = [i for i in gt.keys() if i not in predicted_identifiers]
    assert len(not_in_gt) == 0, f'There are identifiers in the prediction that are not in the GT. Cannot run script.\nNot in gt: {not_in_gt}'
    if len(not_in_pred) != 0:
        print(f'WARNING! Not all identifiers from the ground truth are found in the prediction. This can be intentional or not. GT: {len(gt.keys())}, pred: {len(predicted_identifiers)} identifiers')
    errors = {i: list() for i in all_landmarks}
    detailed_results = {}
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
            dist = np.linalg.norm([i - j for i, j in zip(pred_coords, gt_coords)])
            # if dist > 30:
            #     import IPython;IPython.embed()
            detailed_results[k][ki_gt] = float(np.round(dist, decimals=5))
            errors[ki_gt].append(dist)
    mre_by_landmark = {k: np.mean(errors[k]) for k in errors.keys()}
    mre = np.mean(list(mre_by_landmark.values()))
    save_json({
        'MRE': mre,
        'MRE_by_landmark': {i: float(np.round(mre_by_landmark[i], decimals=5)) for i in mre_by_landmark.keys()},
        'detailed_results': detailed_results
    }, join(folder_with_pred_jsons, 'summary_voxel.json'), sort_keys=False)

def evaluate_MRE_mm(folder_with_pred_jsons: str, gt_json: str, spacing_json: str):
    """
    Computes MRE in millimeters using IMAGE spacing per case.
    Structure mirrors evaluate_MRE, but distances are scaled by spacing.
    Writes: summary_mm.json
    """
    # same filtering as voxel version to only consider per-case voxel JSONs
    predicted_jsons = [i for i in subfiles(folder_with_pred_jsons, suffix='.json', join=False)
                       if i not in ('summary.json', 'summary_voxel.json', 'summary_mm.json', 'dataset.json', 'plans.json', 'combined_predictions.json',
                                    'prediction_all_landmark_voxel.json', 'prediction_all_landmark_mm.json', 'predict_from_raw_data_args.json')
                       and not i.endswith('_mm.json')]

    name_label_dict = load_json(os.path.join(os.path.dirname(gt_json), 'name_to_label.json'))
    all_landmarks = name_label_dict.keys()

    gt = load_json(gt_json)
    spacing_by_case = load_spacing_map(spacing_json)

    predicted_identifiers = [i[:-5] for i in predicted_jsons]
    not_in_gt = [i for i in predicted_identifiers if i not in gt.keys()]
    not_in_pred = [i for i in gt.keys() if i not in predicted_identifiers]
    assert len(not_in_gt) == 0, (f'There are identifiers in the prediction that are not in the GT. '
                                 f'Cannot run script.\nNot in gt: {not_in_gt}')
    if len(not_in_pred) != 0:
        print(f'WARNING! Not all identifiers from the ground truth are found in the prediction. '
              f'This can be intentional or not. GT: {len(gt.keys())}, pred: {len(predicted_identifiers)} identifiers')

    errors = {i: list() for i in all_landmarks}
    detailed_results = {}
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
            # distance in mm: scale per-axis by spacing
            dx = (pred_coords[0] - gt_coords[0]) * sx
            dy = (pred_coords[1] - gt_coords[1]) * sy
            dz = (pred_coords[2] - gt_coords[2]) * sz
            dist_mm = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            detailed_results[k][ki_gt] = float(np.round(dist_mm, decimals=5))
            errors[ki_gt].append(dist_mm)

    mre_by_landmark = {k: np.mean(errors[k]) for k in errors.keys()}
    mre = np.mean(list(mre_by_landmark.values()))
    save_json({
        'MRE': float(mre),
        'MRE_by_landmark': {i: float(np.round(mre_by_landmark[i], decimals=5)) for i in mre_by_landmark.keys()},
        'detailed_results': detailed_results
    }, join(folder_with_pred_jsons, 'summary_mm.json'), sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnLandmark predictions (voxel + mm).")
    parser.add_argument(
        "--nnUNet_raw",
        type=Path,
        required=True,
        help="Path to nnUNet_raw folder, e.g. /path/to/nnUNet_raw/"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (folder under nnUNet_raw), e.g. Dataset732_Afids"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to validation prediction folder"
    )
    args = parser.parse_args()

    dataset_root = args.nnUNet_raw / args.dataset_name

    # load jsons 
    name_to_label_path = dataset_root / "name_to_label.json"
    spacing_json_path = dataset_root / "spacing.json"

    n2l = load_json(name_to_label_path)
    label_to_name = {str(v): k for k, v in n2l.items()}
    spacing_by_case = load_spacing_map(spacing_json_path)

    # aggregate voxel prediction
    pred_voxel_by_case = aggregate_predictions_voxel(args.predictions, label_to_name)
    save_json(pred_voxel_by_case, args.predictions / "prediction_all_landmark_voxel.json")

    # evaluate in voxel space
    evaluate_MRE(args.predictions, str(dataset_root / "all_landmarks_voxel.json"))

    # evaluate in mm
    evaluate_MRE_mm(
        args.predictions,
        str(dataset_root / "all_landmarks_voxel.json"),
        str(spacing_json_path)
    )

if __name__ == "__main__":
    main()