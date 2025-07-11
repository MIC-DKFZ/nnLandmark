from typing import Union, Tuple, List

import cc3d
import edt
import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from torch import nn

from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import build_point, gaussian_kernel_3d, \
    paste_tensor_optionalMax
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.fp_oversampling.oversample_fp import \
    MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25


class ConvertSegToLandmarkTarget(BasicTransform):
    def __init__(self,
                 n_landmarks: int,
                 target_type: str = 'EDT',
                 gaussian_sigma: float = 5,
                 edt_radius: int = 15,
                 ):
        super().__init__()
        self.target_type = target_type
        self.gaussian_sigma = gaussian_sigma
        self.edt_radius = edt_radius
        self.n_landmarks = n_landmarks
        assert target_type in ['Gaussian', 'EDT']

    def apply(self, data_dict, **params):
        seg = data_dict['segmentation']

        # seg must be (1, x, y, z)
        assert len(seg.shape) == 3 or seg.shape[0] == 1
        if len(seg.shape) == 4:
            seg = seg[0]

        regr_target = torch.zeros((self.n_landmarks, *seg.shape[-3:]), dtype=torch.float32, device=seg.device)
        assert seg.ndim == 4, f'this is only implemented for 3d and axes c, x, y, z. Got shape {seg.shape}'

        components = torch.unique(seg)
        components = [i for i in components if i != 0]

        if len(components) > 0:
            stats = cc3d.statistics(seg.numpy().astype(np.uint8))
            for ci in components:
                bbox = stats['bounding_boxes'][ci]  # (slice(3, 9, None), slice(4, 10, None), slice(6, 12, None))
                crop = (seg[bbox] == ci).numpy()
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
                regr_target[ci - 1] = paste_tensor_optionalMax(regr_target[ci - 1], target, insert_bbox, use_max=True)
        # it would be nicer to write that into regression_target but that would require to change the nnunet dataloader so nah
        data_dict['segmentation'] = regr_target
        return data_dict


class nnLandmark_trainer(MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.min_motor_distance = 15
        self.num_epochs = 5

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
        ret: ComposeTransforms = super().get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label
        )
        ret.transforms[-2] = ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), 'EDT',
                                                        edt_radius=15)
        return ret

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_output_channels -= 1
        net = nnUNetTrainer.build_network_architecture(architecture_class_name, arch_init_kwargs,
                                                       arch_init_kwargs_req_import, num_input_channels,
                                                       num_output_channels, enable_deep_supervision)
        return net

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> BasicTransform:
        ret: ComposeTransforms = super().get_validation_transforms(deep_supervision_scales, is_cascaded,
                                                                   foreground_labels, regions, ignore_label)
        ret.transforms[-2] = ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), 'EDT',
                                                        edt_radius=15)
        return ret


    def perform_actual_validation(self, save_probabilities: bool = False):
        pass