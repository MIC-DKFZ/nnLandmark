import multiprocessing
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import SimpleITK as sitk
import blosc2
import numpy as np
import torch
from torch.nn.functional import interpolate
from batchgenerators.utilities.file_and_folder_operations import (
    join, maybe_mkdir_p, isfile, load_json, save_json,
)

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import (
    convert_coordinates, generate_segmentation,
)
from nnunetv2.utilities.helpers import empty_cache


# ───────────────────────────── workers ────────────────────────────── #

@torch.inference_mode()
def process_image(
    image: np.ndarray,
    edge_length: int = 512,
    device: torch.device = torch.device("cuda:0"),
) -> np.ndarray:
    """Resize, clip, rescale → uint8 np.ndarray (keeps GPU off when done)."""
    zoom_factor = edge_length / max(image.shape)
    new_shape = [round(i * zoom_factor) for i in image.shape]
    image = torch.from_numpy(image).to(device, non_blocking=True).float()
    image = interpolate(image[None, None], new_shape, mode="area")[0, 0]
    empty_cache(device)

    mn = torch.kthvalue(image.view(-1), round(0.001 * image.numel()) + 1)[0]
    mx = torch.kthvalue(image.view(-1), round(0.999 * image.numel()) + 1)[0]
    image.clip_(mn, mx)

    image -= image.min()
    image /= image.max()
    image.mul_(255).round_()
    out = image.byte().cpu().numpy()
    empty_cache(device)
    return out


def process_identifier(iden, downloaded_data_dir, imagesTr, labelsTr, edge_length: int = 512):
    """Heavy work executed in worker processes – returns all metadata."""
    try:
        blosc2.set_nthreads(4)
        did, tomoid = iden.split("__")

        json_p = join(downloaded_data_dir, "data", did, f"{tomoid}.json")
        img_p  = join(downloaded_data_dir, "data", did, f"{tomoid}.b2nd")
        labels = load_json(json_p)
        image  = blosc2.open(urlpath=img_p, mode="r")
        motor_annotations = labels["annotation"]

        orig_shape = image.shape
        img_arr    = process_image(image[:], edge_length, torch.device("cuda:0"))
        coords     = convert_coordinates(motor_annotations, orig_shape, img_arr.shape)
        seg        = generate_segmentation(img_arr.shape, coords, max(1, round(6 * edge_length / 512)))

        sitk.WriteImage(sitk.GetImageFromArray(img_arr), join(imagesTr, f"{iden}_0000.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(seg),     join(labelsTr,  f"{iden}.nii.gz"))

        return iden, coords, motor_annotations, orig_shape
    except Exception as e:
        empty_cache(torch.device("cuda:0"))
        print(e)
        raise e


# ───────────────────────────── helpers ────────────────────────────── #

def atomic_save(obj, fn):
    """Write JSON atomically (temp-rename) so crashes never corrupt files."""
    tmp = fn + ".tmp"
    save_json(obj, tmp, sort_keys=False)
    os.replace(tmp, fn)


def load_checkpoint_or_empty(fn):
    try:
        return load_json(fn)
    except Exception:
        return {}


def iden_complete(iden, imgs_dir, lbls_dir, nc, oc, os_):
    """True if we already have both files AND all three dictionaries filled."""
    return (
        (iden in nc and iden in oc and iden in os_)
        and isfile(join(imgs_dir, f"{iden}_0000.nii.gz"))
        and isfile(join(lbls_dir,  f"{iden}.nii.gz"))
    )


# ────────────────────────────── main ─────────────────────────────── #

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    downloaded_data_dir = "/home/isensee/temp/kaggle_byu_additional_data"
    labels_csv          = "/home/isensee/temp/labels.csv"

    target_dataset_name = "Dataset181_Kaggle2025_BYU_FlagMot_BartleysData"
    out_dir             = "/home/isensee/temp"

    imagesTr = join(out_dir, target_dataset_name, "imagesTr")
    labelsTr = join(out_dir, target_dataset_name, "labelsTr")
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # ── identifiers ──────────────────────────────────────────────── #
    loaded_labels = np.loadtxt(labels_csv, skiprows=1, delimiter=",", dtype=str)
    tomos     = loaded_labels[:, 3]
    datasets  = loaded_labels[:, 4]
    identifiers = np.unique([f"{d}__{t}" for d, t in zip(datasets, tomos)]).tolist()
    # identifiers.remove("10230__mba2011-07-18-1")          # still excluded
    print(f"Total unique identifiers: {len(identifiers)}")

    # ── restore checkpoints (if any) ─────────────────────────────── #
    ck_dir = join(out_dir, target_dataset_name)
    ck_new   = join(ck_dir, "train_coordinates.json")
    ck_orig  = join(ck_dir, "train_coordinates_forOrigShapes.json")
    ck_shape = join(ck_dir, "train_OrigShapes.json")

    new_coords   = load_checkpoint_or_empty(ck_new)
    orig_coords  = load_checkpoint_or_empty(ck_orig)
    orig_shapes  = load_checkpoint_or_empty(ck_shape)

    # ── decide what’s left to do ─────────────────────────────────── #
    todo = [
        iden for iden in identifiers
        if not iden_complete(iden, imagesTr, labelsTr, new_coords, orig_coords, orig_shapes)
    ]
    if '10230__mba2011-07-18-1' in todo:
        todo.remove('10230__mba2011-07-18-1')
    if not todo:
        print("Nothing left to do – dataset already complete.")
    else:
        print(f"{len(todo)} / {len(identifiers)} identifiers still missing – processing …")

        blosc2.set_nthreads(8)          # prevent CPU oversubscription inside workers
        with ProcessPoolExecutor(max_workers=6) as ex:
            futures = {
                ex.submit(
                    process_identifier, iden,
                    downloaded_data_dir, imagesTr, labelsTr, 512
                ): iden for iden in todo
            }

            while len(futures) > 0:
                done = {i:v for i, v in futures.items() if i.done()}
                for d in done.keys():
                    if d.exception():
                        print(f"Iden {futures[d]} failed with exception:", d.exception())
                        del futures[d]
                        continue
                    del futures[d]
                    iden, coords, motor_ann, oshape = d.result()

                    new_coords[iden] = [[int(x) for x in c] for c in coords]
                    orig_coords[iden] = [[int(x) for x in c] for c in motor_ann]
                    orig_shapes[iden] = oshape

                    atomic_save(new_coords, ck_new)
                    atomic_save(orig_coords, ck_orig)
                    atomic_save(orig_shapes, ck_shape)
                    print(f"✓ {iden}")

    # ── final dataset.json ───────────────────────────────────────── #
    n_motors = [len(v) for v in orig_coords.values()]
    generate_dataset_json(
        join(out_dir, target_dataset_name),
        {0: "cryoET"},
        {"background": 0, **{f"motor_{i}": i for i in range(1, max(n_motors) + 1)}},
        len(identifiers),
        ".nii.gz",
        citation=None,
        dataset_name=target_dataset_name,
        reference="https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/569921",
        release=None,
        license="Not defined, but fine to use",
        converted_by="Fabian Isensee",
    )
    print("All done ✔")

