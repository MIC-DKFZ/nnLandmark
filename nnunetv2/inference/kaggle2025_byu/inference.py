
###############
# just torch.argwhere
###############

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from batchgenerators.utilities.file_and_folder_operations import subdirs, join, save_json
from ndindex.slice import default
from torch.nn.functional import interpolate

from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates, load_jpgs
from nnunetv2.inference.kaggle2025_byu.gaussian_blur_3d import GaussianBlur3D
from nnunetv2.inference.kaggle2025_byu.iterative_maxpool import iterative_3x3_same_padding_pool3d
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache


@torch.inference_mode()
def resize_image(image: np.ndarray, edge_length: int, device: torch.device) -> torch.Tensor:
    zoom = edge_length / max(image.shape)
    new_shape = [round(s * zoom) for s in image.shape]
    t = torch.from_numpy(image).to(device).float()
    t = interpolate(t[None, None], new_shape, mode='area')[0, 0]
    t = torch.clip(torch.round(t), 0, 255).byte()
    empty_cache(device)
    return t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',
                   default="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test")
    p.add_argument('--output-file',
                   default="/kaggle/working/submission.csv",
                   help="output-file path")
    p.add_argument('--ckpt-dir')
    p.add_argument(
        '--fold',
        type=ast.literal_eval,
        help="tuple of fold identifiers, e.g. \"('all',)\" or (0,1,2)"
    )
    p.add_argument('--threshold', type=float)
    p.add_argument('--gaussian_sigma', type=float, help='only relevant if multiple motor detections '
                                                        'allowed (--allow_multiple_motors), increase if a single'
                                                        ' motor was detected '
                                                        'multiple times in close proximity', default=3)
    p.add_argument('--min_dist', type=int, default=9, help='increase if many motors are detected in proximity')
    p.add_argument('--edge', type=int, default=512)
    p.add_argument('--gpu-id', type=int, default=0,
                   help="This process’s GPU index (0 to num_gpus-1)")
    p.add_argument('--num-gpus', type=int, default=1,
                   help="Total number of GPUs being used")
    p.add_argument('--output_json',
                   default=None,
                   required=False,
                   help="output json file, optional")
    p.add_argument('--allow_multiple_motors', action='store_true', default=False,
                   help='Set this to allow multiple motors to be detected in each tomogram. Use --gaussian_sigma '
                        'and --min_dist to account for too many/few detections in proximity to each other. Defaults '
                        'should work fine, hopefully')
    return p.parse_args()

def main():
    args = parse_args()
    DEVICE = torch.device(f"cuda:{args.gpu_id}")

    # ensure unique filename per GPU
    out_path = args.output_file

    all_tomos = sorted(subdirs(args.input_dir, join=False))
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
    gb = GaussianBlur3D(args.gaussian_sigma, 4, device=DEVICE)

    results_json = {}

    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_jpgs, join(args.input_dir, tomos[0]))
        for i, tomo in enumerate(tomos):
            print(tomo)
            img_np = future.result()
            if i + 1 < len(tomos):
                future = executor.submit(load_jpgs, join(args.input_dir, tomos[i+1]))

            orig_shape = img_np.shape
            img = resize_image(img_np, args.edge, DEVICE).float()
            img = (img - img.mean()) / img.std()

            print(f'Predicting tomo of shape {img.shape}')
            out = pred.predict_logits_from_preprocessed_data(img[None], out_device=DEVICE).float()[None]
            out = torch.sigmoid(out)[0, 0]

            results_json[tomo] = {'motors': [], 'pred_motor_likelihood': []}
            if args.allow_multiple_motors:
                smooth = gb.apply(out[None, None])[0, 0]
                peaks = iterative_3x3_same_padding_pool3d(smooth[None, None], args.min_dist)[0, 0]
                det = (smooth == peaks) & (out > args.threshold)
                coords = torch.argwhere(det)
                ps = [out[tuple(c)].item() for c in coords]
            else:
                coords = torch.argwhere((out == torch.max(out)) & (out > args.threshold))
                ps = [out[tuple(c)].item() for c in coords]

            if len(ps) == 0:
                results.append({'tomo_id': tomo,
                                'Motor axis 0': -1,
                                'Motor axis 1': -1,
                                'Motor axis 2': -1})
            else:
                if args.allow_multiple_motors:
                    for p, c in zip(ps, coords):
                        xyz = convert_coordinates([c.tolist()], img.shape, orig_shape)[0]
                        results.append({'tomo_id': tomo,
                                        'Motor axis 0': xyz[0],
                                        'Motor axis 1': xyz[1],
                                        'Motor axis 2': xyz[2]})
                        results_json[tomo]['motors'].append(xyz)
                        results_json[tomo]['pred_motor_likelihood'].append(p)
                else:
                    # all motors equally likely, pick first
                    best = coords[0].tolist()
                    xyz = convert_coordinates([best], img.shape, orig_shape)[0]
                    results.append({'tomo_id': tomo,
                                    'Motor axis 0': xyz[0],
                                    'Motor axis 1': xyz[1],
                                    'Motor axis 2': xyz[2]})
                    results_json[tomo]['motors'].append(xyz)
                    results_json[tomo]['pred_motor_likelihood'].append(ps[0])
            print('Found motors')
            print(results_json[tomo])

            # free up memory
            del img, out, coords, ps, img_np
            empty_cache(DEVICE)

    # write out clean CSV
    df = pd.DataFrame(results, columns=['tomo_id','Motor axis 0','Motor axis 1','Motor axis 2'])
    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    if args.output_json is not None:
        save_json(results_json, args.output_json, sort_keys=False)


if __name__ == "__main__":
    main()
    # python inference.py --input-dir /home/isensee/temp/kaggle_mult_motors/dev_mult --output-file /home/isensee/temp/kaggle_mult_motors/out2.csv --ckpt-dir /home/isensee/drives/checkpoints/nnUNet_results_kaggle2025_byu/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256 --fold "('all',)" --threshold 0.15 --output_json /home/isensee/temp/kaggle_mult_motors/out2.json --allow_multiple_motors
