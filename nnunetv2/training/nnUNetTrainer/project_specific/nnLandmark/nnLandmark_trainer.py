import multiprocessing, os
from copy import deepcopy
from time import sleep
from typing import Union, Tuple, List

import SimpleITK as sitk
import cc3d
import edt
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd, insert_crop_into_image
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, subfiles
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.helpers.scalar_type import sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform, \
    BrightnessAdditiveTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.intensity.inversion import InvertImageTransform
from batchgeneratorsv2.transforms.intensity.random_clip import CutOffOutliersTransform
from batchgeneratorsv2.transforms.local.brightness_gradient import BrightnessGradientAdditiveTransform
from batchgeneratorsv2.transforms.local.local_gamma import LocalGammaTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform
from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.rot90 import Rot90Transform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.transpose import TransposeAxesTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform, OneOfTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from threadpoolctl import threadpool_limits
from torch import nn, autocast, topk
from torch.nn import functional as F, BCEWithLogitsLoss
from torch.nn.functional import interpolate

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.configuration import default_num_processes
from nnunetv2.dataset_conversion.Dataset119_ToothFairy2_All import load_json
from nnunetv2.dataset_conversion.Dataset737_convert_to_spheres import generate_segmentation
from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import build_point, gaussian_kernel_3d, \
    paste_tensor_optionalMax
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.fp_oversampling.oversample_fp import \
    MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import \
    _brightnessadditive_localgamma_transform_scale, _brightness_gradient_additive_max_strength, _local_gamma_gamma
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import dummy_context, empty_cache


def evaluate_MRE(folder_with_pred_jsons: str, gt_json: str):
    """
    IMPORTANT this function only computes the MRE for all landmarks in the GT.
    It DOES NOT evaluate landmark detection, so whether landmarks are predicted that are not in the GT! I will just
    take the coordinate of each landmark, irrespective of its predicted likelihood.
    So this function can only be used for datasets where all landmarks are present in all images!

    If this is not the case, a more sophisticated evaluation scheme is needed where we evaluate MRE and a landmark detection metric

    TODO this script currently only considers pixel distances and does not take into account the voxel spacing!
    """
    # folder_with_pred_jsons = '/home/isensee/drives/checkpoints/nnUNet_results/Dataset737_FPOSE/nnLandmark_trainer__nnUNetResEncUNetLPlans__3d_fullres/crossval_predictions'
    # gt_json = '/home/isensee/drives/E132-Rohdaten/nnUNetv2/Dataset737_FPOSE/all_landmarks_voxel.json'
    predicted_jsons = [i for i in subfiles(folder_with_pred_jsons, suffix='.json', join=False) if i != 'summary.json']
    # we always predict something for all landmarks, so we can infer how many landmarks there are from any model output json
    name_label_dict = load_json(os.path.join(os.path.dirname(gt_json), 'name_to_label.json'))
    all_landmarks = name_label_dict.keys() # [int(i) for i in load_json(join(folder_with_pred_jsons, predicted_jsons[0])).keys()]
    
    gt = load_json(gt_json)
    
    predicted_identifiers = [i[:-5] for i in predicted_jsons]
    not_in_gt = [i for i in predicted_identifiers if i not in gt.keys()]
    not_in_pred = [i for i in gt.keys() if i not in predicted_identifiers]
    assert len(not_in_gt) == 0, f'There are identifiers in the prediction that are not in the GT. Cannot run script.\nNot in gt: {not_in_gt}'
    if len(not_in_pred) != 0:
        print(f'WARNING! Not all identifiers from the ground truth are found in the prediction. This can be intentional or not. GT: {len(gt.keys())}, pred: {len(predicted_identifiers)} identifiers')
    errors = {i: list() for i in all_landmarks}
    detailed_results = {}
    for k in gt.keys():
        if k in not_in_pred:
            continue
        gt_here = gt[k]
        pred_here = load_json(join(folder_with_pred_jsons, k + '.json'))
        detailed_results[k] = {}
        for ki_gt in gt_here.keys():
            ki_pred = str(name_label_dict[ki_gt])
            pred_coords = pred_here[ki_pred]['coordinates']
            gt_coords = gt_here[ki_gt]
            dist = np.linalg.norm([i - j for i, j in zip(pred_coords, gt_coords)])
            # if dist > 30:
            #     import IPython;IPython.embed()
            detailed_results[k][ki_gt] = float(np.round(dist, decimals=5))
            errors[ki_gt].append(dist)
    mre_by_landmark = {k: np.mean(errors[k]) for k in errors.keys()}
    mre = np.mean(list(mre_by_landmark.values()))
    save_json({
        'MRE': mre,
        'MRE_by_landmark': {i: float(np.round(mre_by_landmark[i], decimals=5)) for i in mre_by_landmark.keys()},
        'detailed_results': detailed_results
    }, join(folder_with_pred_jsons, 'summary.json'), sort_keys=False)


class nnLandmarkLoader(nnUNetDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    bboxes = []
                    target_structs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        bboxes.append(tmp['bboxes'])
                        target_structs.append(tmp['target_struct'])
                    data_all = torch.stack(images)
                    del images
            return {'data': data_all, 'keys': selected_keys, 'target_struct': target_structs, 'bboxes': bboxes}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


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

        components = torch.unique(seg)
        components = [i for i in components if i != 0]

        # now place gaussian or etd on these coordinates
        if self.target_type == 'EDT':
            target = build_point(tuple([self.edt_radius] * 3), use_distance_transform=True, binarize=False)
        else:
            target = torch.from_numpy(gaussian_kernel_3d(self.gaussian_sigma))
            target /= target.max()

        bboxes = {}

        if len(components) > 0:
            stats = cc3d.statistics(seg.numpy().astype(np.uint8))
            for ci in components:
                bbox = stats['bounding_boxes'][ci]  # (slice(3, 9, None), slice(4, 10, None), slice(6, 12, None))
                crop = (seg[bbox] == ci).numpy()
                dist = edt.edt(crop, black_border=True)
                center = np.unravel_index(np.argmax(dist), crop.shape)
                center = [i + j.start for i, j in zip(center, bbox)]
                insert_bbox = [[i - j // 2, i - j // 2 + j] for i, j in zip(center, target.shape)]
                bboxes[ci.item()] = insert_bbox
                # regr_target[ci - 1] = paste_tensor_optionalMax(regr_target[ci - 1], target, insert_bbox, use_max=True)
        # it would be nicer to write that into regression_target but that would require to change the nnunet dataloader so nah
        del data_dict['segmentation']
        data_dict['bboxes'] = bboxes
        data_dict['target_struct'] = target
        return data_dict


# ******************************************************** LOSS FUNCTIONS ********************************************************


class BCE_topK_loss_landmark(nn.Module):
    def __init__(self, k: RandomScalar = 100):
        super().__init__()
        self.bce = BCEWithLogitsLoss(reduction='none')
        # for topk k must be int with k being the number of elements that are returned. We use k as a percentage here,
        # so k=5 will mean top 5 % of pixels!
        self.k = k
        self.preallocated_dummy_target: torch.Tensor = None

    def forward(self, net_output: torch.Tensor, target_structure: torch.Tensor, bboxes):
        # net_output is b, c, x, y, z
        # target_structure is a list of tensors x, y, z
        # bboxes is a list of dicts mapping an index to a bbox
        if self.preallocated_dummy_target is None:
            self.preallocated_dummy_target = torch.zeros(net_output.shape, device=net_output.device,
                                                         dtype=torch.float32)

        with torch.no_grad():
            self.preallocated_dummy_target.zero_()

            for b in range(net_output.shape[0]):
                for c in range(net_output.shape[1]):
                    # insert into preallocated_dummy_target
                    if c + 1 in bboxes[b].keys():
                        paste_tensor_optionalMax(self.preallocated_dummy_target[b, c], target_structure[b], bboxes[b][c + 1], use_max=False)
                    else:
                        pass

        loss = self.bce(net_output, self.preallocated_dummy_target)
        n = max(1, round(np.prod(loss.shape[-3:]) * sample_scalar(self.k) / 100))
        loss = loss.view((*loss.shape[:2], -1))
        loss = topk(loss, k=n, sorted=False)[0]
        loss = loss.mean()
        return loss
    
class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.preallocated_dummy_target: torch.Tensor = None

    def forward(self, net_output, target_structure, bboxes):
        if (self.preallocated_dummy_target is None or
            self.preallocated_dummy_target.shape != net_output.shape or
            self.preallocated_dummy_target.dtype  != net_output.dtype or
            self.preallocated_dummy_target.device != net_output.device):
            self.preallocated_dummy_target = torch.zeros_like(net_output)

        with torch.no_grad():
            self.preallocated_dummy_target.zero_()
            for b in range(net_output.shape[0]):
                for c in range(net_output.shape[1]):
                    if c + 1 in bboxes[b]:
                        paste_tensor_optionalMax(
                            self.preallocated_dummy_target[b, c],
                            target_structure[b],
                            bboxes[b][c + 1],
                            use_max=False
                        )

        pred = torch.sigmoid(net_output)  # not F.sigmoid
        # NOTE: this averages over all voxels (background-dominated)
        loss = F.mse_loss(pred, self.preallocated_dummy_target)
        return loss

class MSE_topK_loss(nn.Module):
    def __init__(self, k: RandomScalar = 100):
        super().__init__()
        # k is percentage of voxels to keep per (b,c)
        self.k = k
        self.preallocated_dummy_target: torch.Tensor = None

    def _ensure_buffer(self, like: torch.Tensor):
        if (self.preallocated_dummy_target is None or
            self.preallocated_dummy_target.shape != like.shape or
            self.preallocated_dummy_target.dtype  != like.dtype or
            self.preallocated_dummy_target.device != like.device):
            # match dtype/device (important for mixed precision)
            self.preallocated_dummy_target = torch.zeros_like(like)

    def forward(self, net_output: torch.Tensor, target_structure, bboxes):
        # net_output: [B, C, X, Y, Z] (or 2D -> [B, C, H, W])
        self._ensure_buffer(net_output)

        with torch.no_grad():
            self.preallocated_dummy_target.zero_()
            # fill the dummy target inside the provided bboxes
            for b in range(net_output.shape[0]):
                for c in range(net_output.shape[1]):
                    if c + 1 in bboxes[b]:
                        paste_tensor_optionalMax(
                            self.preallocated_dummy_target[b, c],
                            target_structure[b],
                            bboxes[b][c + 1],
                            use_max=False
                        )

        # logits -> probabilities in [0,1] for MSE
        pred = torch.sigmoid(net_output)

        # per-voxel squared error (no reduction)
        per_voxel = (pred - self.preallocated_dummy_target) ** 2  # [B, C, ...]
        B, C = per_voxel.shape[:2]
        spatial = per_voxel.shape[2:]
        N = int(np.prod(spatial))
        n = max(1, round(N * sample_scalar(self.k) / 100))

        # top-k over spatial dims per (b, c)
        per_voxel = per_voxel.view(B, C, -1)
        topk_vals = torch.topk(per_voxel, k=n, dim=-1, sorted=False).values
        loss = topk_vals.mean()
        return loss



# *******************************************************************************************************************************************
# ******************************************************** FABIS NN LANDMARK TRAINER ********************************************************
# *******************************************************************************************************************************************




# This one still has BYU data augmentation
class nnLandmark_trainer(MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.edt_radius = 15
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        disable mirroring
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.75, 1.35))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')

        mirror_axes = None
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

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
        """
        Copied from the motor trainer. We need to disable axes transpose because if messes with left/right landmarks
        """
        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
        transforms = []

        transforms.append(RandomTransform(
            CutOffOutliersTransform(
                (0, 2.5),
                (98.5, 100),
                p_synchronize_channels=1,
                p_per_channel=0.5,
                p_retain_std=0.5
            ), apply_probability=0.2
        ))

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.3, rotation=rotation_for_DA, p_scaling=0.3, scaling=(0.6, 1.67),
                p_synchronize_scaling_across_axes=0.8,
                bg_style_seg_sampling=False , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            transforms.append(RandomTransform(
                Rot90Transform(num_rot_per_combination=(1, 2, 3), num_axis_combinations=(1, 4), allowed_axes=set(valid_axes)),
                apply_probability=0.5
            ))
            # transforms.append(RandomTransform(
            #     TransposeAxesTransform(allowed_axes=set(valid_axes)),
            #     apply_probability=0.5
            # ))

        OneOfTransform([
            RandomTransform(
                MedianFilterTransform(
                (2, 8), p_same_for_each_channel=0.5, p_per_channel=0.5
            ), apply_probability=0.2),
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.3, 1.5),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        )
        ])

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.2),
                p_per_channel=0.5,
                synchronize_channels=True
            ), apply_probability=0.3
        ))

        transforms.append(RandomTransform(
                BrightnessAdditiveTransform(0, 0.5, per_channel=True, p_per_channel=0.5),
                apply_probability=0.1
            ))

        transforms.append(OneOfTransform(
            [
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.75, 1.25)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=0.5
                    ), apply_probability=0.3
                ),
                RandomTransform(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.75, 1.25)),
                        synchronize_channels=False,
                        p_per_channel=0.5
                    ), apply_probability=0.3
                )
            ]
        ))

        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.15
        ))

        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.6, 2)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.2
        ))

        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.6, 2)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.2
        ))

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        # transforms.append(RandomTransform(
        #     BlankRectangleTransform(
        #         [[max(1, p // 10), p // 3] for p in patch_size],
        #         rectangle_value=torch.mean,
        #         num_rectangles=(1, 5),
        #         force_square=False,
        #         p_per_channel=0.5
        #     ),
        #     apply_probability=0.2)
        # )

        transforms.append(RandomTransform(
            BrightnessGradientAdditiveTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                max_strength=_brightness_gradient_additive_max_strength,
                same_for_all_channels=False,
                mean_centered=True,
                clip_intensities=False,
                p_per_channel=0.5
            ),
            apply_probability=0.2))

        transforms.append(RandomTransform(
            LocalGammaTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                _local_gamma_gamma,
                same_for_all_channels=False,
                p_per_channel=0.5
            ), apply_probability=0.2
        ))

        transforms.append(RandomTransform(
            SharpeningTransform(
                (0.1, 1.5),
                p_same_for_each_channel=0.5,
                p_per_channel=0.5,
                p_clamp_intensities=0.5
            ), apply_probability=0.2
        ))

        transforms.append(RandomTransform(
            InvertImageTransform(p_invert_image=1, p_synchronize_channels=0.5, p_per_channel=0.5), apply_probability=0.2
        ))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        transforms.append(ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), 'EDT',
                                                        edt_radius=self.edt_radius))

        transforms = ComposeTransforms(transforms)

        return transforms


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
        transforms: ComposeTransforms = nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded,
                                                                                foreground_labels, regions,
                                                                                ignore_label)
        transforms.transforms.append(ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), 'EDT',
                                                        edt_radius=self.edt_radius))
        return transforms

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        dsj = deepcopy(self.dataset_json)
        n_landmarks = len(self.label_manager.foreground_labels)
        dsj['labels'] = {'background': 0, **{str(i): i for i in range(1, n_landmarks)}}
        # don't worry about use_mirroring=True. self.inference_allowed_mirroring_axes is None.
        # we set perform_everything_on_device=False because landmark tasks often have vram issues because of how many landmarks there are
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=False, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        dsj, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as export_pool:
            worker_list = [i for i in export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers): # enumerate(['tomo_4c1ca8']): #
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)

                # we do [:] to convert blosc2 to numpy
                data = data[:]
                data = torch.from_numpy(data)

                if self.is_cascaded:
                    raise NotImplementedError

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')

                # predict logits
                with torch.no_grad():
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    empty_cache(self.device)
                    prediction = F.sigmoid(prediction).float()

                    # detect landmarks as maximum predicted value in each channel
                    mx = prediction.max(-1)[0].max(-1)[0].max(-1)[0]
                    detected_coords = [torch.argwhere(prediction[c] == mx[c])[0] for c in range(len(mx))]
                    empty_cache(self.device)

                det_p = [prediction[j][*i].item() for j, i in enumerate(detected_coords)]
                detected_coords = [[i.item() for i in j] for j in detected_coords]

                # convert coords to original geometry
                # revert transpose
                detected_coords = [[i[j] for j in self.plans_manager.transpose_backward] for i in detected_coords]
                # revert resize
                new_coordinates = convert_coordinates(detected_coords, data.shape[-3:], properties['shape_after_cropping_and_before_resampling'])
                # revert cropping
                crop_offset = [i[0] for i in properties['bbox_used_for_cropping']]
                new_coordinates = [[k + crop_offset[l] for l, k in enumerate(i)] for i in new_coordinates]
                # alex: change x and z coordinates
                new_coordinates = [[coord[2], coord[1], coord[0]] for coord in new_coordinates]

                # we need to export the coordinates as segs before we invert nibabel reorient
                # export coordinates
                out_dict = {i: {'coordinates': j, 'likelihood': l} for i, j, l in zip(range(1, n_landmarks + 1), new_coordinates, det_p)}

                # generate a segmentation visualizing the predicted coords
                seg = generate_segmentation(properties['shape_before_cropping'], {i: out_dict[i]['coordinates'] for i in out_dict.keys()}, radius=4)
                self.plans_manager.image_reader_writer_class().write_seg(seg, join(validation_output_folder, k + self.dataset_json['file_ending']), properties)

                # If we loaded with nibabel with reorient we need to revert any potential reorientation we did
                if self.plans_manager.image_reader_writer_class == NibabelIOWithReorient:
                    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
                    orig_affine = properties['nibabel_stuff']['original_affine']
                    img_ornt = io_orientation(orig_affine)
                    # Orientation of RAS
                    ras_ornt = axcodes2ornt('RAS')
                    # Transform from original to RAS
                    revert = ornt_transform(ras_ornt, img_ornt)
                    revert[:, 1] = revert[:, 1][::-1]

                    # reconstruct original shape
                    shape_before = [0, 0, 0]
                    for i in range(3):
                        target_axis = int(revert[i, 0])  # axis in the original
                        shape_before[target_axis] = properties['shape_before_cropping'][i]

                    new_coordinates = np.array(new_coordinates)
                    out = np.zeros_like(new_coordinates)
                    for i in range(3):  # loop over output axes
                        orig_axis = int(revert[i, 0])
                        flip = int(revert[i, 1])
                        if flip == 1:
                            out[:, i] = new_coordinates[:, orig_axis]
                        else:
                            out[:, i] = shape_before[orig_axis] - 1 - new_coordinates[:, orig_axis]
                    new_coordinates = out

                # now save coordinates (potentially corrected)
                out_dict = {i: {'coordinates': j, 'likelihood': l} for i, j, l in zip(range(1, n_landmarks + 1), [[int(l) for l in m] for m in new_coordinates], det_p)}
                save_json(out_dict, join(validation_output_folder, k + '.json'))

                # export heatmaps, only works for nifti
                # if 'sitk_stuff' in properties.keys():
                #     # revert resize
                #     prediction_resized = interpolate(prediction[None], properties['shape_after_cropping_and_before_resampling'], mode='trilinear').cpu()[0]
                #     empty_cache(self.device)
                #
                #     # revert cropping
                #     prediction_uncropped = torch.zeros((prediction_resized.shape[0], *properties['shape_before_cropping']), device='cpu', dtype=torch.float)
                #     insert_crop_into_image(prediction_uncropped, prediction_resized, properties['bbox_used_for_cropping'])
                #     del prediction_resized
                #
                #     # revert transpose
                #     prediction_uncropped = prediction_uncropped.numpy().transpose([0] + [i + 1 for i in self.plans_manager.transpose_backward])
                #
                #     # round to 0.01
                #     prediction_uncropped = np.round(prediction_uncropped, decimals=2)
                #
                #     # convert to nifti and export
                #     for i in range(1, prediction_uncropped.shape[0] + 1):
                #         prediction_uncropped_itk = sitk.GetImageFromArray(prediction_uncropped[i-1])
                #         prediction_uncropped_itk.SetSpacing(properties['sitk_stuff']['spacing'])
                #         prediction_uncropped_itk.SetOrigin(properties['sitk_stuff']['origin'])
                #         prediction_uncropped_itk.SetDirection(properties['sitk_stuff']['direction'])
                #
                #         sitk.WriteImage(prediction_uncropped_itk, join(validation_output_folder, k + f'__{i:03d}.nii.gz'))

        evaluate_MRE(validation_output_folder, join(nnUNet_raw, self.plans_manager.dataset_name, 'all_landmarks_voxel.json'))

    def train_step(self, batch: dict) -> dict:
        data = batch['data']

        data = data.to(self.device, non_blocking=True)
        target_structure = [i.to(self.device, non_blocking=True) for i in batch['target_struct']]

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # import IPython;IPython.embed()
            # if False:
            #     from batchviewer import view_batch
            #     view_batch(data[0], target[0][0], F.sigmoid(output[0][0]))

         # take loss out of autocast! Sigmoid is not stable in fp16
        l = self.loss(output, target_structure, batch['bboxes'])

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']

        data = data.to(self.device, non_blocking=True)
        target_structure = [i.to(self.device, non_blocking=True) for i in batch['target_struct']]

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target_structure, batch['bboxes'])

        return {'loss': l.detach().cpu().numpy()}

    def _build_loss(self):
        loss = BCE_topK_loss_landmark(k=20)

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.soft_dice)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        assert not self.enable_deep_supervision, 'bruh.'
        return loss

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnLandmarkLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling,
                                 random_offset=[i // 3 for i in self.configuration_manager.patch_size])
        dl_val = nnLandmarkLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling,
                                  random_offset=[i // 3 for i in self.configuration_manager.patch_size])

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnLandmark_trainer_edt7(nnLandmark_trainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.edt_radius = 7




# *****************************************************************************************************************************************
# ******************************************************** ALEX CHANGED STUFF HERE ********************************************************
# *****************************************************************************************************************************************



# Here now set back to original nnUNet DA
class nnLandmark_fabi(nnLandmark_trainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.blobb_radius = 15
        self.enable_deep_supervision = False
        self.blobb_type = 'EDT'


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
        """
        Set back to the original nnUNet data augmentation. 
        """
        #matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        #valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        transforms.append(ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), self.blobb_type,
                                                        edt_radius=self.blobb_radius))

        transforms = ComposeTransforms(transforms)

        return transforms

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> BasicTransform:
        transforms: ComposeTransforms = nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded,
                                                                                foreground_labels, regions,
                                                                                ignore_label)
        transforms.transforms.append(ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), self.blobb_type,
                                                        edt_radius=self.blobb_radius))
        return transforms


class nnLandmark_v1(nnLandmark_fabi):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.blobb_radius = 4
        self.enable_deep_supervision = False
        self.blobb_type = 'Gaussian'
        self.initial_lr = 5e-4
        self.grad_clip = 1.

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
        """
        Set back to the original nnUNet data augmentation. 
        """
        #matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        #valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        transforms.append(ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), self.blobb_type,
                                                        edt_radius=self.blobb_radius))

        transforms = ComposeTransforms(transforms)

        return transforms

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> BasicTransform:
        transforms: ComposeTransforms = nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded,
                                                                                foreground_labels, regions,
                                                                                ignore_label)
        transforms.transforms.append(ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), self.blobb_type,
                                                        edt_radius=self.blobb_radius))
        return transforms

    def train_step(self, batch: dict) -> dict:
        data = batch['data']

        data = data.to(self.device, non_blocking=True)
        target_structure = [i.to(self.device, non_blocking=True) for i in batch['target_struct']]

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # import IPython;IPython.embed()
            # if False:
            #     from batchviewer import view_batch
            #     view_batch(data[0], target_structure[0][0], F.sigmoid(output[0][0]))

         # take loss out of autocast! Sigmoid is not stable in fp16
        l = self.loss(output, target_structure, batch['bboxes'])

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def _build_loss(self):

        loss = MSE_loss()

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.soft_dice)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        assert not self.enable_deep_supervision, 'bruh.'
        return loss
    
    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)

        return optimizer, lr_scheduler   
