import SimpleITK
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import ConvertSegToRegrTarget
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

import numpy as np
from PIL import Image

def generate_overlay_images(image_3d, float_map, output_dir, max_alpha_strength=1.0, intensity_multiplier=2.5, intensity_reduction_factor=0.7):
    # Ensure the image and float map are numpy arrays
    image_3d = np.array(image_3d)
    float_map = np.array(float_map)

    # Normalize the entire 3D image globally to the range [0, 255]
    image_3d = np.uint8((image_3d - image_3d.min()) / (image_3d.max() - image_3d.min()) * 255)

    # Lime Green (RGB: [0, 255, 0])
    lime_green = np.array([0, 255, 0]) * intensity_multiplier
    lime_green = np.clip(lime_green, 0, 255).astype(np.uint8)

    # Loop through the slices in the 3D image and generate overlays
    for i in range(image_3d.shape[0]):
        # Extract the 2D slice of the image and floating-point map
        image_slice = image_3d[i]
        float_map_slice = float_map[i]

        # Create a blank RGBA image for the overlay
        overlay = np.zeros((image_slice.shape[0], image_slice.shape[1], 4), dtype=np.uint8)

        # Set the RGB channels to the lime green color
        overlay[:, :, :3] = lime_green

        # Scale the floating point map to modulate the alpha channel with configurable maximum strength
        alpha = (float_map_slice * max_alpha_strength * 255).astype(np.uint8)

        # Ensure alpha is within the 0-255 range
        overlay[:, :, 3] = np.clip(alpha, 0, 255)

        # Create the base grayscale image in RGBA format
        base_image = np.dstack([image_slice, image_slice, image_slice, np.ones_like(image_slice) * 255])

        # Reduce the intensity of the base image to allow the color to pop more
        reduced_image = np.clip(base_image[:, :, :3] * intensity_reduction_factor, 0, 255)

        # Perform alpha blending (overlay the lime green with the base image)
        alpha_blend = overlay[:, :, 3] / 255.0
        blended_image = (1 - alpha_blend[:, :, None]) * reduced_image + alpha_blend[:, :, None] * overlay[:, :, :3]

        # Convert the blended image to uint8
        blended_image = np.uint8(blended_image)

        # Create a PIL Image and save it as a PNG
        output_path = f'{output_dir}/overlay_{i:03d}.png'
        Image.fromarray(blended_image).save(output_path)
        # print(f"Saved overlay image: {output_path}")

if __name__ == "__main__":

    dataset_id = 189
    dataset_name = maybe_convert_to_dataset_name(dataset_id)

    base = join(nnUNet_raw, dataset_name)

    dataset = get_filenames_of_train_images_and_targets(base)
    tr = ConvertSegToRegrTarget('EDT', 0, 25)
    out_dir = '/home/isensee/temp/byu_overlays'
    maybe_mkdir_p(out_dir)

    np.random.seed(42)

    for iden in np.random.choice(list(dataset.keys()), 15, replace=False):
        image = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(join(base, 'imagesTr', dataset[iden]['images'][0])))
        labels = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(join(base, 'labelsTr', dataset[iden]['label'])))

        # convert labels to float map
        float_map = tr.apply({'segmentation': torch.from_numpy(labels[None])})['segmentation'][0].numpy()
        maybe_mkdir_p(join(out_dir, iden))
        generate_overlay_images(image,float_map, join(out_dir, iden), 1, intensity_multiplier=1.5, intensity_reduction_factor=0.7)
