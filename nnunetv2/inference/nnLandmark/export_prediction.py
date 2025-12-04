from typing import Union, List
import os, json
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle, join

from nnunetv2.paths import nnUNet_raw
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


@torch.inference_mode()
def convert_probabilities_to_segmentation(predicted_probabilities: torch.Tensor) -> torch.Tensor:
    """
    predicted_probabilities: (C, X, Y[, Z]) after nonlinearity.
    For each channel, keep the 27 highest voxels as segmentation.
    """

    seg = torch.zeros_like(predicted_probabilities[0], dtype=torch.int16)
    C, *spatial = predicted_probabilities.shape
    N = int(torch.tensor(spatial).prod())
    k = min(27, N)

    for ch, cls_id in enumerate(range(C)):
        flat = predicted_probabilities[ch].flatten()
        top_idx = torch.topk(flat, k).indices
        coords = torch.unravel_index(top_idx, spatial)
        seg[coords] = cls_id+1

    return seg


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    if isinstance(predicted_logits, np.ndarray):
            predicted_logits = torch.from_numpy(predicted_logits)
    predicted_probabilities = torch.sigmoid(predicted_logits)
    segmentation = convert_probabilities_to_segmentation(predicted_probabilities)
    del predicted_logits

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties_dict['bbox_used_for_cropping'])
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)

    # revert cropping
    predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                properties_dict[
                                                                                    'bbox_used_for_cropping'],
                                                                                properties_dict[
                                                                                    'shape_before_cropping'])
    predicted_probabilities = predicted_probabilities.cpu().numpy()
    # revert transpose
    predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                        plans_manager.transpose_backward])
    torch.set_num_threads(old_threads)

    return segmentation_reverted_cropping, predicted_probabilities

# ──────────────────────────────────────────────────────────────────────────────
# 1) Replace your extractor with this: returns (coord_tuple_or_None, likelihood)
#    coords are (x, y, z); likelihood is float in [0,1].
# ──────────────────────────────────────────────────────────────────────────────
def _extract_landmark_coord_and_likelihood(probs_zyx: np.ndarray,
                                           top_percent: float = 0.5):
    """
    probs_zyx: (Z, Y, X) ndarray with probabilities in [0,1].
    - Drop exact 1.0 plateaus (set to 0) to avoid degenerate flats.
    - Optionally restrict to the top `top_percent`% positive voxels.
    Returns:
      ( (x, y, z), likelihood_float ) or (None, 0.0) if empty.
    """
    a = np.asarray(probs_zyx, dtype=np.float32, order='C').copy()
    a[a == 1.0] = 0.0  # drop perfect plateaus as in your current logic

    if not np.any(a > 0):
        return None, 0.0

    # Mask to top k% of positive voxels (by value), default 0.5%
    vals = a[a > 0]
    n = vals.size
    k = max(1, int(np.ceil(n * (top_percent / 100.0))))
    if k < n:
        thr = np.partition(vals, n - k)[n - k]
        mask = a >= thr
    else:
        mask = a > 0

    if not np.any(mask):
        # Fall back to simple global argmax
        z, y, x = np.unravel_index(int(np.argmax(a)), a.shape)
        return (int(x), int(y), int(z)), float(a[z, y, x])

    # Argmax within mask
    masked = np.where(mask, a, 0.0)
    z, y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return (int(x), int(y), int(z)), float(a[z, y, x])


# ──────────────────────────────────────────────────────────────────────────────
# 2) Keep your convert_* functions as-is. Only tweak JSON writing below.
#    Assumes `probabilities_final` is (C, Z, Y, X) after all reversals!
# ──────────────────────────────────────────────────────────────────────────────
def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor],
                                  properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str],
                                  output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  num_threads_torch: int = default_num_processes):

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)

    # Convert & resample back to original geometry
    segmentation_final, probabilities_final = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch
    )
    del predicted_array_or_file

    # ── save multi-label segmentation (nnU-Net writer handles header/affine etc.)
    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)

    # ── build flat per-class JSON (keys "1","2",..., no case wrapper), incl. likelihood
    probs = probabilities_final
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # Ensure shape (C, Z, Y, X)
    if probs.ndim == 3:  # (C, Y, X) -> (C, 1, Y, X)
        probs = probs[:, None, ...]

    # Foreground labels are the class ids (usually 1..C)
    class_ids = list(label_manager.foreground_labels)

    out_json = {}
    for ch, cls_id in enumerate(class_ids):
        coord, lik = _extract_landmark_coord_and_likelihood(probs[ch])  # returns (x,y,z), float
        if coord is None:
            out_json[str(int(cls_id))] = {
                "coordinates": [None, None, None],
                "likelihood": 0.0
            }
        else:
            x, y, z = coord
            out_json[str(int(cls_id))] = {
                "coordinates": [int(x), int(y), int(z)],
                "likelihood": float(lik)
            }

    with open(output_file_truncated + ".json", "w") as f:
        json.dump(out_json, f, indent=4)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes,
                      dataset_class=None) \
        -> None:

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if dataset_class is None:
        nnUNetDatasetBlosc2.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    else:
        dataset_class.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    torch.set_num_threads(old_threads)
