import SimpleITK as sitk
from typing import Tuple, List
import pandas as pd
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.morphology.morphology_helper import generate_ball
from batchgenerators.utilities.file_and_folder_operations import join, nifti_files, subfiles, save_json
import numpy as np
import concurrent.futures
import os

from nnunetv2.dataset_conversion.Dataset119_ToothFairy2_All import load_json
from nnunetv2.dataset_conversion.kaggle_byu.additional_external_data.create_nnunet_dataset import get_coords_from_seg
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def generate_segmentation(shape, coordinates: dict[int, List[int]], radius: int = 2):
    sphere = generate_ball([radius] * 3, dtype=np.uint8)
    seg = np.zeros(shape, dtype=np.uint8)

    for lb, ci in coordinates.items():
        bbox = [[i - radius, i + radius + 1] for i in ci]
        insert_crop_into_image(seg, sphere * lb, bbox)

    return seg


def process_case(args):
    lmf, base_dir, landmark_name_to_id, train_identifiers = args
    identifier = lmf[:-14]
    is_train = identifier in train_identifiers
    landmarks = load_json(join(base_dir, 'landmarks', lmf))['landmarks']
    lm_dict = {landmark_name_to_id[i]: [j[1], j[2], j[0]] for i, j in landmarks.items()}

    img_path = join(base_dir, 'imagesTr' if is_train else 'imagesTs', identifier + '_0000.nii.gz')
    img_itk = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img_itk)

    seg = generate_segmentation(img_np.shape, lm_dict, radius=4)
    seg_itk = sitk.GetImageFromArray(seg)
    seg_itk.SetSpacing(img_itk.GetSpacing())
    seg_itk.SetOrigin(img_itk.GetOrigin())
    seg_itk.SetDirection(img_itk.GetDirection())

    out_path = join(base_dir, 'labelsTr' if is_train else 'labelsTs', identifier + '.nii.gz')
    sitk.WriteImage(seg_itk, out_path)

    return identifier, lm_dict


if __name__ == '__main__':
    source_datset_id = 737
    dataset_name = maybe_convert_to_dataset_name(source_datset_id)
    base_dir = join(nnUNet_raw, dataset_name)

    dsj = load_json(join(base_dir, 'dataset.json'))
    json_files = subfiles(join(base_dir, 'landmarks'), join=False)

    # discover all possible landmarks
    possible_landmarks = set(j for i in json_files for j in load_json(join(base_dir, 'landmarks', i))['landmarks'].keys())
    assignment = {i: j for i, j in zip(range(1, len(possible_landmarks) + 1), possible_landmarks)}
    landmark_name_to_id = {j: i for i, j in assignment.items()}

    train_identifiers = [i[:-12] for i in nifti_files(join(base_dir, 'imagesTr'), join=False)]

    all_landmark_coords = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_case, [(lmf, base_dir, landmark_name_to_id, train_identifiers) for lmf in json_files])
        for identifier, lm_dict in results:
            all_landmark_coords[identifier] = lm_dict

    save_json(all_landmark_coords, join(base_dir, 'landmark_coordinates.json'))

    # update dataset.json
    del dsj['training']
    del dsj['test']

    dsj['labels'] = {'background': 0, **landmark_name_to_id}
    save_json(dsj, join(base_dir, 'dataset.json'), sort_keys=False)
