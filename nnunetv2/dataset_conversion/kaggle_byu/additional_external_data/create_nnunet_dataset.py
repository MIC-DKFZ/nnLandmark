import shutil

import SimpleITK as sitk
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, load_json, nifti_files, isfile

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates
from nnunetv2.paths import nnUNet_raw


def get_coords_from_seg(file: str):
    img = sitk.GetArrayFromImage(sitk.ReadImage(file))
    labels = list(np.sort(pd.unique(img.ravel())))
    labels.remove(0)
    coords = []
    for lb in labels:
        com = np.argwhere(img == lb).mean(0)
        coords.append([round(i) for i in com])
    return coords, img.shape


if __name__ == '__main__':

    dataset_name = 'KaggleBYU_NonBartleyData'
    dataset_id = 188
    dataset_name_full = f'Dataset{dataset_id:03d}_{dataset_name}'

    out_base = join(nnUNet_raw, dataset_name_full)
    imagesTr = join(out_base, 'imagesTr')
    labelsTr = join(out_base, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    source_dir_base = '/media/isensee/T9/non_bartley_data_to_add'
    source_dir_img = join(source_dir_base, 'data')
    source_dir_labels = join(source_dir_base, 'labels')
    source_dir_corr = join(source_dir_base, 'corrected')
    uncertain_cases = [i[:-12] for i in load_json(join(source_dir_corr, 'uncertain_cases.json'))]

    cases = [i[:-12] for i in nifti_files(source_dir_img, join=False)]
    n = 0
    for c in cases:
        if c in uncertain_cases:
            print('not using case {}'.format(c))
            continue
        else:
            n+=1
            shutil.copy(join(source_dir_img, f'{c}_0000.nii.gz'), imagesTr)
            if isfile(join(source_dir_corr, f'{c}.nii.gz')):
                shutil.copy(join(source_dir_corr, f'{c}.nii.gz'), labelsTr)
                print('used corrected label for case {}'.format(c))
            else:
                shutil.copy(join(source_dir_labels, f'{c}.nii.gz'), labelsTr)

    # we just set max motor to 10, should be enough. Too lazy.
    generate_dataset_json(
        out_base,
        {0: "cryoET"},
        {"background": 0, **{f"motor_{i}": i for i in range(1, 10 + 1)}},
        n,
        ".nii.gz",
        citation=None,
        dataset_name=dataset_name_full,
        reference="Data downloaded with script from https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/569921. Only includes data that is NOT already present in Bartley's collection!",
        release=None,
        license="Same as source datasets, probably CC0",
        converted_by="Fabian Isensee",
    )

    # generate train_coordinates.json, train_coordinates_forOrigShapes.json, train_OrigShapes.json
    cases = [i[:-7] for i in nifti_files(labelsTr, join=False)]
    original_shapes = {}
    train_shapes = {}
    train_coordinates = {}
    train_train_coordinates_orig_shapes = {}
    original_shapes_all = load_json('/media/isensee/T9/non_bartley_data/original_shapes.json')

    for c in cases:
        coords, shape = get_coords_from_seg(join(labelsTr, f'{c}.nii.gz'))

        original_shapes[c] = original_shapes_all[c]
        train_shapes[c] = shape

        train_coordinates[c] = coords
        train_train_coordinates_orig_shapes[c] = convert_coordinates(coords, shape, original_shapes[c])

    save_json(original_shapes, join(out_base, 'train_OrigShapes.json'))
    save_json(train_train_coordinates_orig_shapes, join(out_base, 'train_coordinates_forOrigShapes.json'))
    save_json(train_coordinates, join(out_base, 'train_coordinates.json'))
    save_json(train_shapes, join(out_base, 'train_shapes.json'))
