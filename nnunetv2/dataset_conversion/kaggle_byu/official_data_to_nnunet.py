import gc

import SimpleITK as sitk
import matplotlib.image as mpimg
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.morphology.morphology_helper import generate_ball
from batchgenerators.utilities.file_and_folder_operations import *
from torch.nn.functional import interpolate

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.helpers import empty_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


@torch.inference_mode()
def resize_image(image: np.ndarray, edge_length: int = 512, device: torch.device = torch.device('cuda:0')) -> np.array:
    zoom_factor = edge_length / max(image.shape)
    new_shape = [round(i * zoom_factor) for i in image.shape]

    image_torch = torch.from_numpy(image).to(device).float()
    image_torch = interpolate(image_torch[None, None], new_shape, mode='area')[0, 0]
    image_out = torch.clip(torch.round(image_torch), min=0, max=255).byte().cpu().numpy()

    # cleanup
    del image_torch
    gc.collect()
    empty_cache(device)

    return image_out


def load_jpgs(folder: str):
    jpgs = subfiles(folder, suffix='.jpg')
    image = np.vstack([mpimg.imread(i)[None] for i in jpgs])
    return image


def convert_coordinates(coordinates_in_original, original_shape, new_shape):
    new_coords = []
    for ci in coordinates_in_original:
        new_coords.append([round(c * (n - 1) / (o - 1)) for c, n, o in zip(ci, new_shape, original_shape)])
    return new_coords


def generate_segmentation(shape, coordinates, radius: int = 2):
    sphere = generate_ball([radius] * 3, dtype=np.uint8)
    seg = np.zeros(shape, dtype=np.uint8)
    for lb, ci in enumerate(coordinates):
        bbox = [[i - radius, i + radius + 1] for i in ci]
        insert_crop_into_image(seg, sphere * (lb + 1), bbox)
    return seg


def get_coordinates_from_labels(identifier, labels: np.ndarray):
    cols = np.where(labels[:, 1] == identifier)[0]
    if len(cols) > 0:
        coords = labels[cols, 2:5].astype(float)
        if len(coords) == 1 and all([i == -1 for i in coords[0]]):
            coords = []
    else:
        coords = None
    return coords


if __name__ == '__main__':
    """

    """
    nnunet_dataset_id = 184
    task_name = "Kaggle2025_BYU_FlagellarMotors_384"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    base = '/media/isensee/raw_data/bact_motors/official'
    labels = np.loadtxt(join(base, 'train_labels.csv'), delimiter=',', skiprows=1, dtype=str)

    executor = ThreadPoolExecutor(max_workers=1)
    writer_executor = ProcessPoolExecutor(max_workers=8)

    futures = []


    def write_nifti(image, path):
        sitk.WriteImage(sitk.GetImageFromArray(image), path)


    def prefetch_image(identifier):
        img = load_jpgs(join(base, 'train', identifier))
        return identifier, img


    identifiers = subdirs(join(base, 'train'), join=False)
    it = iter(identifiers)
    iden = next(it)
    future = executor.submit(prefetch_image, iden)

    new_coords = {}
    orig_coords = {}
    orig_shapes = {}

    for next_iden in it:
        iden, image = future.result()
        future = executor.submit(prefetch_image, next_iden)

        print(iden)
        orig_shape = image.shape
        image = resize_image(image, 384, torch.device('cuda:0'))
        coords_orig = get_coordinates_from_labels(iden, labels)
        coords_reshaped = convert_coordinates(coords_orig, orig_shape, image.shape)

        new_coords[iden] = coords_reshaped
        orig_coords[iden] = coords_orig
        orig_shapes[iden] = orig_shape

        seg = generate_segmentation(image.shape, coords_reshaped, 6)

        futures.append(writer_executor.submit(write_nifti, image, join(imagestr, iden + '_0000.nii.gz')))
        futures.append(writer_executor.submit(write_nifti, seg, join(labelstr, iden + '.nii.gz')))

    # process the final one
    iden, image = future.result()
    print(iden)
    orig_shape = image.shape
    image = resize_image(image, 384, torch.device('cuda:0'))
    coords_orig = get_coordinates_from_labels(iden, labels)
    coords_reshaped = convert_coordinates(coords_orig, orig_shape, image.shape)

    new_coords[iden] = coords_reshaped
    orig_coords[iden] = coords_orig
    orig_shapes[iden] = orig_shape

    seg = generate_segmentation(image.shape, coords_reshaped, 6)
    futures.append(writer_executor.submit(write_nifti, image, join(imagestr, iden + '_0000.nii.gz')))
    futures.append(writer_executor.submit(write_nifti, seg, join(labelstr, iden + '.nii.gz')))

    # Wait for all NIfTI writing to complete
    for f in as_completed(futures):
        f.result()  # to propagate any exceptions

    for k in identifiers:
        for i in range(len(new_coords[k])):
            new_coords[k][i] = [int(i) for i in new_coords[k][i]]
        orig_coords[k] = [[int(i) for i in j] for j in orig_coords[k]]


    save_json(new_coords, join(out_base, 'train_coordinates.json'), sort_keys=False)
    save_json(orig_coords, join(out_base, 'train_coordinates_forOrigShapes.json'), sort_keys=False)
    save_json(orig_shapes, join(out_base, 'train_OrigShapes.json'), sort_keys=False)

    n_motors = [len(v) for v in orig_coords.values()]
    generate_dataset_json(out_base, {0: 'cryoET'}, {'background': 0, **{f'motor_{i}': i for i in range(1, max(n_motors) + 1)}},
                          len(identifiers), '.nii.gz', citation=None, dataset_name=task_name,
                          reference='https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/data',
                          release=None, license='MIT', converted_by='Fabian Isensee')

    # test set, we only have 3 cases, no annotation
    identifiers = subdirs(join(base, 'test'), join=False)
    orig_shapes = {}
    for iden in identifiers:
        print(iden)
        image = load_jpgs(join(base, 'test', iden))
        orig_shape = image.shape
        image = resize_image(image, 512, torch.device('cuda:0'))
        orig_shapes[iden] = orig_shape

        sitk.WriteImage(sitk.GetImageFromArray(image), join(imagests, iden + '_0000.nii.gz'))
    save_json(orig_shapes, join(out_base, 'test_OrigShapes.json'), sort_keys=False)

    # i determined the radius for the segmentation based off of that
    # min_dist = 99999
    # for c in new_coords.keys():
    #     n = len(new_coords[c])
    #     if n >= 2:
    #         for i in range(n):
    #             for j in range(i + 1, n):
    #                 dist = np.linalg.norm(np.array(new_coords[c][i]) - np.array(new_coords[c][j]))
    #                 if dist < min_dist:
    #                     min_dist = dist