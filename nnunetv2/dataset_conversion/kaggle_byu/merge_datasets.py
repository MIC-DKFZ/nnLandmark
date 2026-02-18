from copy import deepcopy
import os
del os.environ['nnUNet_preprocessed']
del os.environ['nnUNet_results']

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

if __name__ == '__main__':
    current_nnunet_raw = os.environ['nnUNet_raw']
    if current_nnunet_raw.endswith('/'):
        current_nnunet_raw = current_nnunet_raw[:-1]

    source_datasets = [187, 190]
    target_datset_name = 'Dataset191_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_withFP'

    out_base = join(nnUNet_raw, target_datset_name)
    maybe_mkdir_p(out_base)

    dataset = {}
    targets = {}
    targets_orig = {}
    shp_orig = {}

    for sd in source_datasets:
        raw_dir = join(nnUNet_raw, maybe_convert_to_dataset_name(sd))
        lb = load_json(join(raw_dir, 'train_coordinates.json'))
        lb_orig = load_json(join(raw_dir, 'train_coordinates_forOrigShapes.json'))
        shp = load_json(join(raw_dir, 'train_OrigShapes.json'))
        ds = get_filenames_of_train_images_and_targets(raw_dir)
        for k, v in ds.items():
            dataset[k] = {
                'images': [i.replace(current_nnunet_raw, '$nnUNet_raw') for i in ds[k]['images']],
                'label': ds[k]['label'].replace(current_nnunet_raw, '$nnUNet_raw')
            }
        targets.update(lb)
        targets_orig.update(lb_orig)
        shp_orig.update(shp)

    max_motors = 0
    for v in targets.values():
        max_motors = max(max_motors, len(v))

    source_dsjs = [load_json(join(nnUNet_raw, maybe_convert_to_dataset_name(i), 'dataset.json')) for i in source_datasets]
    target_dsj = deepcopy(source_dsjs[0])
    target_dsj['labels'] = {
        'background': 0,
        **{f'motor_{i:02d}': i for i in range(1, max_motors + 1)}
    }
    target_dsj['name'] = target_datset_name[len('Dataset182_'):]
    target_dsj['numTraining'] = len(targets)
    target_dsj['license'] = f'See source datasets {source_datasets}'
    target_dsj['reference'] = [s['reference'] for s in source_dsjs]
    target_dsj['dataset'] = dataset
    save_json(target_dsj, join(out_base, 'dataset.json'), sort_keys=False)

    save_json(targets, join(out_base, 'train_coordinates.json'))
    save_json(targets_orig, join(out_base, 'train_coordinates_forOrigShapes.json'))
    save_json(shp_orig, join(out_base, 'train_OrigShapes.json'))
