import shutil

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


if __name__ == '__main__':
    np.random.seed(42)

    pred_dir = '/media/isensee/T9/non_bartley_data_predicted'
    tomo_ids = [i[:-7] for i in subfiles(pred_dir, suffix='.nii.gz', join=False)]
    predicted_motors = {}
    for jsonfile in subfiles(pred_dir, suffix='.json', join=False):
        predicted_motors[jsonfile[:-5]] = load_json(join(pred_dir, jsonfile))
    tomos_with_motors = [i for i in tomo_ids if len(predicted_motors[i]['coordinates']) > 0]
    tomos_without_motors = [i for i in tomo_ids if i not in tomos_with_motors]

    selected = set()
    # draw 250 tomos with motors
    selected.update([str(i) for i in np.random.choice(tomos_with_motors, size=250, replace=False)])

    remaining = [i for i in tomos_without_motors if i not in selected]
    dataset_ids = np.unique([i.split('__')[0] for i in remaining]) # 86 datasets

    # draw a max of 4 per dataset
    for d in dataset_ids:
        ds_cases = [i for i in remaining if i.split('__')[0] == d]
        drawn_cases = [str(i) for i in np.random.choice(ds_cases, size=min(len(ds_cases), 4), replace=False)]
        [remaining.remove(i) for i in drawn_cases]
        selected.update(drawn_cases)

    data_dir = '/media/isensee/T9/non_bartley_data'
    out_base = '/media/isensee/T9/non_bartley_data_to_add'
    out_data = join(out_base, 'data')
    out_labels = join(out_base, 'labels')

    maybe_mkdir_p(out_data)
    maybe_mkdir_p(out_labels)

    for s in selected:
        shutil.copy(join(pred_dir, s + '.nii.gz'), out_labels)
        shutil.copy(join(pred_dir, s + '.json'), out_labels)
        shutil.copy(join(data_dir, s + '_0000.nii.gz'), out_data)

    # make sure all datasets are represented with motors
    remaining_with_motors = [i for i in tomos_with_motors if i not in selected]
    newly_selected = set()
    for d in dataset_ids:
        cases_with_motors_ds = [i for i in selected if i.split('__')[0] == d and i in tomos_with_motors]
        tomos_with_motors_ds = [i for i in tomos_with_motors if i.split('__')[0] == d]

        print(d, len(cases_with_motors_ds), len(tomos_with_motors_ds))

        if len(cases_with_motors_ds) < 4:
            remaining_with_motors_ds = [i for i in remaining_with_motors if i in tomos_with_motors and i.split('__')[0] == d]
            add = min(4 - len(cases_with_motors_ds), len(remaining_with_motors_ds))
            if add > 0:
                newly_selected.update([str(i) for i in np.random.choice(remaining_with_motors_ds, size=add, replace=False)])

    for s in newly_selected:
        shutil.copy(join(pred_dir, s + '.nii.gz'), out_labels)
        shutil.copy(join(pred_dir, s + '.json'), out_labels)
        shutil.copy(join(data_dir, s + '_0000.nii.gz'), out_data)