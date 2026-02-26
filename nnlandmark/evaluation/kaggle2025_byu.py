from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from typing import List
import numpy as np


def compute_f_beta(
    predicted_coordinates: List[np.ndarray],
    gt_coordinates: List[np.ndarray],
    beta: int = 2,
    pixel_dist_threshold: float = 35.0,
) -> float:
    """
    Computes the F-beta score between predicted and ground truth 3D coordinates.

    This function is designed for evaluating object localization in 3D microscopy images.
    It compares predicted coordinates against ground truth annotations using a distance threshold
    to determine true positives, false positives, and false negatives.

    Each predicted point is considered a true positive (TP) if it is within `pixel_dist_threshold`
    of any ground truth point. Multiple predictions may match a single ground truth point
    without penalty. This is intentional and appropriate when images are expected to contain
    a single object in test time. Predictions that do not match any ground truth are counted
    as false positives (FP), and ground truth points with no matching prediction are counted
    as false negatives (FN).

    Args:
        predicted_coordinates (List[np.ndarray]): A list of arrays (one per image), each of shape (N, 3),
            containing the predicted 3D coordinates. Use an empty array if there are no predictions for an image.
        gt_coordinates (List[np.ndarray]): A list of arrays (one per image), each of shape (M, 3),
            containing the ground truth 3D coordinates. Use an empty array if there are no GTs for an image.
        beta (int, optional): Weight of recall in the harmonic mean. Default is 2 (F2 score).
        pixel_dist_threshold (float, optional): Maximum distance (in pixels) to consider a prediction
            as matching a ground truth point. Default is 35, which corresponds roughly to 1000 Å on Dataset142.

    Returns:
        float: The computed F-beta score across all images.

    Notes:
        - This metric assumes independent evaluation per image.
        - Multiple predictions matching the same GT are not penalized.
        - FP is incremented for unmatched predictions.
        - FN is incremented for unmatched GT points.
    """
    assert len(predicted_coordinates) == len(gt_coordinates)
    TP = 0
    FP = 0
    FN = 0

    for preds, gts in zip(predicted_coordinates, gt_coordinates):
        if len(gts) == 0:
            FP += len(preds)
            continue
        if len(preds) == 0:
            FN += len(gts)
            continue

        distances = np.linalg.norm(preds[:, np.newaxis, :] - gts[np.newaxis, :, :], axis=2)
        thresholded = distances < pixel_dist_threshold
        TP += np.sum(np.any(thresholded, 0))
        FN += np.sum(~np.any(thresholded, 0))
        FP += np.sum(~np.any(thresholded, 1))

    denom = (1 + beta ** 2) * TP + beta ** 2 * FN + FP
    if denom == 0:
        return 0.0
    return (1 + beta ** 2) * TP / denom


def evaluate_folder(folder, gt_file, MIN_P=0.1):
    gt = load_json(gt_file)
    pred_files = [i for i in subfiles(folder, join=False, suffix='.json') if not i.startswith('scores')]
    pred_keys = [i[:-5] for i in pred_files]
    gt_list = [np.array(gt[k]) for k in pred_keys]
    preds = [load_json(join(folder, k)) for k in pred_files]

    probs = np.linspace(MIN_P, 1, num=200)  # np.unique([j for i in preds for j in i['probabilities']])
    fbeta = []
    for threshold in probs:
        pred_list_here = []
        for i in range(len(preds)):
            pred_list_here.append(np.array([preds[i]['coordinates'][j] for j in range(len(preds[i]['probabilities'])) if
                                            preds[i]['probabilities'][j] > threshold]))
        # we need to threshold the prediction
        fbeta.append(compute_f_beta(pred_list_here, gt_list, 2, 35))

    import seaborn as sns
    # Plot using seaborn
    sns.set(style="whitegrid", context="talk")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=probs, y=fbeta, marker='o')
    plt.xlabel("Probability Threshold", fontsize=14)
    plt.ylabel("F\u03B2 Score (β=2)", fontsize=14)
    plt.title("F\u03B2 Score vs Probability Threshold", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(join(folder, 'f2_vs_threshold.png'))

    fbeta_2 = []
    probs2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in probs2:
        pred_list_here = []
        for i in range(len(preds)):
            pred_list_here.append(np.array([preds[i]['coordinates'][j] for j in range(len(preds[i]['probabilities'])) if
                                            preds[i]['probabilities'][j] > threshold]))
        # we need to threshold the prediction
        fbeta_2.append(compute_f_beta(pred_list_here, gt_list, 2, 35))

    save_json(
        {
            'f2_max': np.max(fbeta),
            'f2_max_at_threshold': probs[fbeta == np.max(fbeta)][0],
            'f2_at_0.5': fbeta_2[4],
            'rough_sweep': {i: j for i, j in zip(probs2, fbeta_2)},
        }, join(folder, 'scores.json')
    )
    save_json(
        {
            'precise_sweep': {i: j for i, j in zip(probs, fbeta)},
        }, join(folder, 'scores_precise.json')
    )