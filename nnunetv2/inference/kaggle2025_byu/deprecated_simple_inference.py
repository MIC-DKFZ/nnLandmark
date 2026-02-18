from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, subdirs, join
from torch.nn.functional import interpolate
from torch.nn import functional as F

from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates, load_jpgs
from nnunetv2.inference.kaggle2025_byu.gaussian_blur_3d import GaussianBlur3D
from nnunetv2.inference.kaggle2025_byu.iterative_maxpool import iterative_3x3_same_padding_pool3d
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache


@torch.inference_mode()
def resize_image(image: np.ndarray, edge_length: int = 512, device: torch.device = torch.device('cuda:0')) -> np.array:
    zoom_factor = edge_length / max(image.shape)
    new_shape = [round(i * zoom_factor) for i in image.shape]
    image = torch.from_numpy(image).to(device).float()
    image = interpolate(image[None, None], new_shape, mode='area')[0, 0]
    image = torch.clip(torch.round(image), min=0, max=255).byte()
    empty_cache(device)
    return image


if __name__ == '__main__':
    # is torch.compile active?

    INPUT_DIR = '/media/isensee/raw_data/bact_motors/byu-locating-bacterial-flagellar-motors-2025/test'
    OUTPUT_FILE = '/media/isensee/raw_data/bact_motors/byu-locating-bacterial-flagellar-motors-2025/test_results.csv'
    CHECKPOINT_DIR = '/media/isensee/data/results_nnUNet_remake/Dataset142_Kaggle2025_BYU_FlagellarMotors/MotorRegressionTrainer_MSELoss__nnUNetPlans__3d_fullres'
    FOLD = (0, )
    DEVICE = torch.device('cuda', 0)
    THRESHOLD = 0.317
    MIN_MOTOR_DIST = 13

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

    pred.initialize_from_trained_model_folder(
        CHECKPOINT_DIR, FOLD
    )

    # hack to make it generate just one output channel
    pred.label_manager._all_labels = [0]
    gb = GaussianBlur3D(2, 3, device=DEVICE)

    # Preload first image manually
    tomo_list = subdirs(INPUT_DIR, join=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_jpgs, join(INPUT_DIR, tomo_list[0]))

        with open(OUTPUT_FILE, 'w') as f:
            f.write('tomo_id,Motor axis 0,Motor axis 1,Motor axis 2\n')
            with torch.no_grad():
                for i, tomo in enumerate(tomo_list):
                    # Wait for the background-loaded image
                    img: np.ndarray = future.result()

                    # Start loading the next image (if any)
                    if i + 1 < len(tomo_list):
                        future = executor.submit(load_jpgs, join(INPUT_DIR, tomo_list[i + 1]))

                    orig_shape = img.shape
                    img: torch.Tensor = resize_image(img, 512, DEVICE).float()
                    img -= img.mean()
                    img /= img.std()
                    resized_shape = img.shape

                    out = pred.predict_logits_from_preprocessed_data(img[None], out_device=DEVICE).float()
                    del img
                    out = F.sigmoid(out[None])[0, 0]

                    smooth_pred = gb.apply(out[None, None])[0, 0]
                    mp = iterative_3x3_same_padding_pool3d(smooth_pred[None, None], MIN_MOTOR_DIST)[0, 0]
                    detections = (smooth_pred == mp) & (out > THRESHOLD)
                    detected_coords = torch.argwhere(detections)

                    det_p = [out[*i].item() for i in detected_coords]
                    detected_coords = [[i.item() for i in j] for j in detected_coords]

                    if len(detected_coords) == 0:
                        f.write(f'{tomo},-1,-1,-1\n')
                    else:
                        most_likely = detected_coords[np.argmax(det_p)]
                        most_likely = convert_coordinates([most_likely], resized_shape, orig_shape)[0]
                        f.write(f'{tomo},{most_likely[0]},{most_likely[1]},{most_likely[2]}\n')

                    del out, smooth_pred, mp, detections, det_p, detected_coords
                    empty_cache(DEVICE)
