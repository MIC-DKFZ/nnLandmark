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

class ConvertSegToRegrTarget(BasicTransform):
    def __init__(self,
                 target_type: str = 'Gaussian',
                 gaussian_sigma: float = 5,
                 edt_radius: int = 10
                 ):
        super().__init__()
        self.target_type = target_type
        self.gaussian_sigma = gaussian_sigma
        self.edt_radius = edt_radius
        assert target_type in ['Gaussian', 'EDT']

    def apply(self, data_dict, **params):
        seg = data_dict['segmentation']
        regr_target = torch.zeros_like(seg, dtype=torch.float32)
        assert seg.ndim == 4, f'this is only implemented for 3d and axes c, x, y, z. Got shape {seg.shape}'
        for c in range(seg.shape[0]):
            components = torch.unique(seg[c])
            components = [i for i in components if i != 0]
            if len(components) > 0:
                stats = cc3d.statistics(seg[c].numpy().astype(np.uint8))
                for ci in components:
                    bbox = stats['bounding_boxes'][ci]  # (slice(3, 9, None), slice(4, 10, None), slice(6, 12, None))
                    crop = (seg[c][bbox] == ci).numpy()
                    dist = edt.edt(crop, black_border=True)
                    center = np.unravel_index(np.argmax(dist), crop.shape)
                    center = [i + j.start for i, j in zip(center, bbox)]
                    # now place gaussian or etd on these coordinates
                    if self.target_type == 'EDT':
                        target = build_point(tuple([self.edt_radius] * 3), use_distance_transform=True, binarize=False)
                    else:
                        target = torch.from_numpy(gaussian_kernel_3d(self.gaussian_sigma))
                        target /= target.max()
                    insert_bbox = [[i - j // 2, i - j // 2 + j] for i, j in zip(center, target.shape)]
                    regr_target[c] = paste_tensor_optionalMax(regr_target[c], target, insert_bbox, use_max=True)
        # it would be nicer to write that into regression_target but that would require to change the nnunet dataloader so nah
        data_dict['segmentation'] = regr_target
        return data_dict



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

def gaussian_kernel_2d(sigma, truncate=3.0):
    """
    Generate a 2D Gaussian kernel.

    Args:
        sigma (float or tuple): Standard deviation(s) of the Gaussian.
                                If scalar → same for both axes.
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        kernel (np.ndarray): 2D Gaussian kernel, normalized to sum=1.
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)

    # kernel size for each axis (odd for symmetry)
    size = [int(truncate * s + 0.5) * 2 + 1 for s in sigma]

    y, x = [np.arange(-sz // 2 + 1, sz // 2 + 1) for sz in size]
    yy, xx = np.meshgrid(y, x, indexing='ij')

    kernel = np.exp(-(xx ** 2 / (2 * sigma[0] ** 2) +
                      yy ** 2 / (2 * sigma[1] ** 2)))
    kernel /= kernel.sum()

    # replicate into 3 channels: (3, H, W);
    kernel_3ch = np.repeat(kernel[None, :, :], 3, axis=0)
    return kernel_3ch


def paste_tensor_optionalMax(target, source, bbox, use_max=False):
    """
    Safely paste `source` into `target` using a bounding box `bbox`.
    Automatically clips if the bbox goes outside the target.

    Supports:
        - 2D: H x W
        - 2D with channel: 1 x H x W
        - 3D: D x H x W
        - 3D with channel: 1 x D x H x W

    Args:
        target: np.ndarray or torch.Tensor
        source: same type as target
        bbox: list of [start, end] per dimension
        use_max: bool, whether to combine using max instead of overwriting
    """
    import numpy as np
    import torch

    is_numpy = isinstance(target, np.ndarray)
    xp = np if is_numpy else torch

    target_shape = target.shape
    ndim = len(bbox)

    # Detect if first dimension is channel=1
    has_channel = target.ndim == ndim + 1 and target.shape[0] == 1

    target_slices = []
    source_slices = []

    for i, (b0, b1) in enumerate(bbox):
        t_start = max(b0, 0)
        t_end = min(b1, target_shape[i + (1 if has_channel else 0)])
        if t_start >= t_end:
            # No overlap in this dim -> skip
            return target

        s_start = t_start - b0
        s_end = s_start + (t_end - t_start)

        # Always use slice objects
        target_slices.append(slice(t_start, t_end))
        source_slices.append(slice(s_start, s_end))

    if has_channel:
        target_slices = [slice(0, 1)] + target_slices
        source_slices = [slice(0, 1)] + source_slices

    t_slice = tuple(target_slices)
    s_slice = tuple(source_slices)

    # Compute shapes safely
    shape_target = [t.stop - t.start for t in target_slices]
    shape_source = [s.stop - s.start for s in source_slices]

    # Skip if any dimension has zero length
    if any(st <= 0 for st in shape_target) or any(ss <= 0 for ss in shape_source):
        return target

    # Paste or max
    if use_max:
        target[t_slice] = xp.maximum(target[t_slice], source[s_slice])
    else:
        target[t_slice] = source[s_slice]

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
