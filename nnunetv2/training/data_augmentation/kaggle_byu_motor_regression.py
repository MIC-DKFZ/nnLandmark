from functools import lru_cache

import SimpleITK
import cc3d
import edt
import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk, ball


@lru_cache(maxsize=5)
def build_point(radii, use_distance_transform, binarize):
    max_radius = max(radii)
    ndim = len(radii)

    # Create a spherical (or circular) structuring element with max_radius
    if ndim == 2:
        structuring_element = disk(max_radius)
    elif ndim == 3:
        structuring_element = ball(max_radius)
    else:
        raise ValueError("Unsupported number of dimensions. Only 2D and 3D are supported.")

    # Convert the structuring element to a tensor
    structuring_element = torch.from_numpy(structuring_element.astype(np.float32))

    # Create the target shape based on the sampled radii
    target_shape = [round(2 * r + 1) for r in radii]

    if any([i != j for i, j in zip(target_shape, structuring_element.shape)]):
        structuring_element_resized = torch.nn.functional.interpolate(
            structuring_element.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions for interpolation
            size=target_shape,
            mode='trilinear' if ndim == 3 else 'bilinear',
            align_corners=False
        )[0, 0]  # Remove batch and channel dimensions after interpolation
    else:
        structuring_element_resized = structuring_element

    if use_distance_transform:
        # Convert the structuring element to a binary mask for distance transform computation
        binary_structuring_element = (structuring_element_resized >= 0.5).numpy()

        # Compute the Euclidean distance transform of the binary structuring element
        structuring_element_resized = distance_transform_edt(binary_structuring_element)

        # Normalize the distance transform to have values between 0 and 1
        structuring_element_resized /= structuring_element_resized.max()
        structuring_element_resized = torch.from_numpy(structuring_element_resized)

    if binarize and not use_distance_transform:
        # Normalize the resized structuring element to binary (values near 1 are treated as the point region)
        structuring_element_resized = (structuring_element_resized >= 0.5).float()
    return structuring_element_resized




@lru_cache(maxsize=2)
def gaussian_kernel_3d(sigma, truncate=3.0):
    """
    Generate a 3D Gaussian kernel.

    Args:
        sigma (float or tuple): Standard deviation of the Gaussian.
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        kernel (np.ndarray): 3D Gaussian kernel.
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)

    # Determine kernel size (odd for symmetry)
    size = [int(truncate * s + 0.5) * 2 + 1 for s in sigma]
    z, y, x = [np.arange(-sz // 2 + 1, sz // 2 + 1) for sz in size]
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    kernel = np.exp(-(xx ** 2 / (2 * sigma[0] ** 2) +
                      yy ** 2 / (2 * sigma[1] ** 2) +
                      zz ** 2 / (2 * sigma[2] ** 2)))
    kernel /= kernel.sum()
    return kernel


def paste_tensor_optionalMax(target: torch.Tensor, source: torch.Tensor, bbox, use_max=False):
    """
    Paste or combine a source tensor into a target tensor using a given bounding box,
    with optional pixelwise maximum.

    Args:
        target (torch.Tensor): 3D tensor of shape (T0, T1, T2) on CPU.
        source (torch.Tensor): 3D tensor of shape (S0, S1, S2) on CPU.
                        paste_tensor_optionalMax       Must match the size of bbox.
        bbox (list or tuple): Bounding box as [[x1, x2], [y1, y2], [z1, z2]]
                              in target coordinate space.
        use_max (bool): If True, combine using pixelwise max instead of direct paste.

    Returns:
        torch.Tensor: Modified target tensor.
    """
    target_shape = target.shape
    target_indices = []
    source_indices = []

    for i, (b0, b1) in enumerate(bbox):
        t_start = max(b0, 0)
        t_end = min(b1, target_shape[i])
        if t_start >= t_end:
            return target  # No overlap

        s_start = t_start - b0
        s_end = s_start + (t_end - t_start)

        target_indices.append((t_start, t_end))
        source_indices.append((s_start, s_end))

    # Extract slices
    tz0, tz1 = target_indices[0]
    ty0, ty1 = target_indices[1]
    tx0, tx1 = target_indices[2]
    sz0, sz1 = source_indices[0]
    sy0, sy1 = source_indices[1]
    sx0, sx1 = source_indices[2]

    target_slice = target[tz0:tz1, ty0:ty1, tx0:tx1]
    source_slice = source[sz0:sz1, sy0:sy1, sx0:sx1]

    if use_max:
        target[tz0:tz1, ty0:ty1, tx0:tx1] = torch.maximum(target_slice, source_slice)
    else:
        target[tz0:tz1, ty0:ty1, tx0:tx1] = source_slice

    return target


if __name__ == '__main__':
    case = 'tomo_00e463'
    image = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(f'/media/isensee/raw_data/nnUNet_raw/Dataset142_Kaggle2025_BYU_FlagellarMotors/imagesTr/{case}_0000.nii.gz')))[None]
    seg = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(f'/media/isensee/raw_data/nnUNet_raw/Dataset142_Kaggle2025_BYU_FlagellarMotors/labelsTr/{case}.nii.gz')))[None]
    t = ConvertSegToRegrTarget('EDT', 5, 25)
    ret = t(image=image, segmentation=seg)
    from batchviewer import view_batch
    view_batch(255*ret['segmentation'] + ret['image'], ret['segmentation'], ret['image'])
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(ret['segmentation'].numpy()[0]), f'/media/isensee/raw_data/nnUNet_raw/Dataset142_Kaggle2025_BYU_FlagellarMotors/{case}_edt25.nii.gz')
