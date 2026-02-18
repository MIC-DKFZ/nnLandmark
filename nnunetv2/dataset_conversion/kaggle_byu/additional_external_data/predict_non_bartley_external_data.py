###############
# THIS IS WHAT WE HAVE BEEN USING SO FAR
###############

import argparse
import ast
import os
from concurrent.futures import ThreadPoolExecutor

import SimpleITK
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, nifti_files, maybe_mkdir_p, \
    isfile, save_json
from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import generate_segmentation
from nnunetv2.inference.kaggle2025_byu.gaussian_blur_3d import GaussianBlur3D
from nnunetv2.inference.kaggle2025_byu.iterative_maxpool import iterative_3x3_same_padding_pool3d
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache

def load_case(filename):
    return SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(filename))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',
                   default="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test")
    p.add_argument('--output-dir',
                   help="output directory")
    p.add_argument('--ckpt-dir')
    p.add_argument(
        '--fold',
        type=ast.literal_eval,
        help="tuple of fold identifiers, e.g. ('all',) or (0,1,2)"
    )
    p.add_argument('--threshold', type=float)
    p.add_argument('--min-dist', type=int, default=13)
    p.add_argument('--edge', type=int, default=512)
    p.add_argument('--gpu-id', type=int, default=0,
                   help="This process’s GPU index (0 to num_gpus-1)")
    p.add_argument('--num-gpus', type=int, default=1,
                   help="Total number of GPUs being used")
    return p.parse_args()

@torch.inference_mode()
def main():
    os.environ['nnUNet_compile'] = 't'
    args = parse_args()
    DEVICE = torch.device(f"cuda:0")

    # ensure unique filename per GPU
    out_path = args.output_dir
    maybe_mkdir_p(out_path)

    all_tomos = sorted(nifti_files(args.input_dir, join=False))
    tomos = all_tomos[args.gpu_id::args.num_gpus]

    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=DEVICE,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    pred.initialize_from_trained_model_folder(args.ckpt_dir, args.fold)
    pred.label_manager._all_labels = [0]
    gb = GaussianBlur3D(2, 3, device=DEVICE)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_case, join(args.input_dir, tomos[0]))
        for i, tomo in enumerate(tomos):
            img_np = future.result()
            outfile = join(out_path, os.path.basename(tomos[i][:-12] + '.nii.gz'))
            if i + 1 < len(tomos):
                future = executor.submit(load_case, join(args.input_dir, tomos[i+1]))

            if isfile(outfile):
                continue

            img = torch.from_numpy(img_np).float()
            img = (img - img.mean()) / img.std()

            logits = pred.predict_logits_from_preprocessed_data(img[None], out_device=DEVICE).float()[None]
            out = torch.sigmoid(logits)[0, 0]
            smooth = gb.apply(out[None, None])[0, 0]
            peaks = iterative_3x3_same_padding_pool3d(smooth[None, None], args.min_dist)[0, 0]
            det = (smooth == peaks) & (out > args.threshold)
            coords = torch.argwhere(det)
            coords = [[i.item() for i in j] for j in coords]
            ps = [out[tuple(c)].item() for c in coords]

            if len(coords) == 0:
                seg = generate_segmentation(img.shape, coords, radius=6).astype(np.uint8)
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(seg), outfile)
            else:
                seg = generate_segmentation(img.shape, coords, radius=6).astype(np.uint8)
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(seg), outfile)

            save_json({'coordinates': coords, 'probabilities': ps},
                      join(out_path, os.path.basename(tomos[i][:-12] + '.json')))

            # free up memory
            del img, logits, out, smooth, peaks, det, coords, ps
            empty_cache(DEVICE)


if __name__ == "__main__":
    main()
    # python predict_non_bartley_external_data.py --input-dir /media/isensee/T9/non_bartley_data --output-dir /media/isensee/T9/non_bartley_data_predicted --ckpt-dir /home/isensee/drives/checkpoints/nnUNet_results_kaggle2025_byu/Dataset186_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartley_corrected/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3kep__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_224_224 --fold "('all',)" --threshold 0.08 --gpu-id 40 --num-gpus 50
