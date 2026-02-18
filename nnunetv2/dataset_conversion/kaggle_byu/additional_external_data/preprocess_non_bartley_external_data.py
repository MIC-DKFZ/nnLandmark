import os
import json
import blosc2
import torch
from concurrent.futures import ProcessPoolExecutor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, maybe_mkdir_p, subdirs, subfiles
from nnunetv2.dataset_conversion.kaggle_byu.bartley_additional_data import process_image
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import SimpleITK as sitk


def process_file(base, ds, img, ref_keys, target_dir, dparams, mmap_kwargs):
    identifier = f"{ds}__{img[:-5]}"
    out_path = join(target_dir, identifier + '_0000.nii.gz')

    # Load raw data and record shape
    data = blosc2.open(urlpath=join(base, ds, img), mode='r', dparams=dparams, **mmap_kwargs)
    orig_shape = data.shape

    # If already in reference dataset, skip processing but return shape
    if identifier in ref_keys:
        print(identifier, "already exists in reference dataset")
        return identifier, orig_shape

    # If output already exists, skip heavy processing but still record shape
    if os.path.exists(out_path):
        print(identifier, "output already exists, skipping processing")
        return identifier, orig_shape

    proc = None
    try:
        # Process image
        arr = data[:]  # load into memory
        proc = process_image(arr, 512, torch.device('cpu'))
    except KeyboardInterrupt:
        print(identifier, "interrupted during processing, writing partial result")
    finally:
        if proc is not None:
            # Always write image if we have a processed array
            sitk.WriteImage(sitk.GetImageFromArray(proc), out_path)
    return identifier, orig_shape


if __name__ == '__main__':
    base = '/home/isensee/temp/kaggle_byu_additional_data/data'
    ref_dataset = load_json(join(nnUNet_raw, maybe_convert_to_dataset_name(186), 'dataset.json'))
    target_dir = '/media/isensee/T9/non_bartley_data'
    maybe_mkdir_p(target_dir)

    dparams = {'nthreads': 8}
    mmap_kwargs = {'mmap_mode': 'r'}
    torch.set_num_threads(8)

    # Prepare tasks
    ref_keys = set(ref_dataset['dataset'].keys())
    tasks = [(base, ds, img, ref_keys, target_dir, dparams, mmap_kwargs)
             for ds in subdirs(base, join=False)
             for img in subfiles(join(base, ds), suffix='.b2nd', join=False)]

    # Process in parallel and collect shapes
    shapes = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_file, *task) for task in tasks]
        for future in futures:
            identifier, orig_shape = future.result()
            shapes[identifier] = orig_shape

    # Save shapes dict to JSON
    shapes_path = join(target_dir, 'original_shapes.json')
    with open(shapes_path, 'w') as f:
        json.dump({k: list(v) for k, v in shapes.items()}, f)
    print(f"Saved original shapes for {len(shapes)} files to {shapes_path}")
