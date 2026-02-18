from typing import Union, Tuple, List

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import ConvertSegToRegrTarget
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.losses.bce_topk import \
    MotorRegressionTrainer_BCEtopK20Loss


class MotorRegressionTrainer_BCEtopK20Loss_gaussian(MotorRegressionTrainer_BCEtopK20Loss):
    def get_training_transforms(
            self, patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms: ComposeTransforms = nnUNetTrainer.get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales,
                                                                              mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm,
                                                                              is_cascaded, foreground_labels, regions, ignore_label)
        assert isinstance(transforms.transforms[-1], DownsampleSegForDSTransform)
        transforms.transforms.pop(-1)

        transforms.transforms[0].mode_seg = 'nearest'
        transforms.transforms.append(ConvertSegToRegrTarget('Gaussian', gaussian_sigma=self.min_motor_distance // 3,
                                                            edt_radius=self.min_motor_distance))
        transforms.transforms.append(DownsampleSegForDSTransform(deep_supervision_scales))

        return transforms