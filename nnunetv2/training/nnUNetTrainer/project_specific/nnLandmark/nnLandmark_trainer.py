import multiprocessing
from copy import deepcopy
from time import sleep
from typing import Union, Tuple, List

import cc3d
import edt
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from threadpoolctl import threadpool_limits
from torch import nn, autocast, topk
from torch.nn import functional as F, BCEWithLogitsLoss

from nnunetv2.configuration import default_num_processes, ANISO_THRESHOLD
from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import build_point, gaussian_kernel_3d, \
    paste_tensor_optionalMax
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.fp_oversampling.oversample_fp import \
    MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.losses.bce_topk import BCE_topK_loss
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import dummy_context

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


class nnLandmark_trainer(MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.min_motor_distance = 15
        self.num_epochs = 20
        self.enable_deep_supervision = False

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
        del ret.transforms[-1]
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
        transforms: ComposeTransforms = nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded,
                                                                                foreground_labels, regions,
                                                                                ignore_label)
        transforms.transforms.append(ConvertSegToLandmarkTarget(len(self.label_manager.foreground_labels), 'EDT',
                                                        edt_radius=15))
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
        dsj['labels'] = {'background': 0, **{str(i): i for i in range(1, 22)}}
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
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
                output_filename_truncated = join(validation_output_folder, k)

                # predict logits
                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = F.sigmoid(prediction).float()

                # detect landmarks as maximum predicted value in each channel
                mx = prediction.max(-1)[0].max(-1)[0].max(-1)[0]
                detected_coords = [torch.argwhere(prediction[c] == mx[c])[0] for c in range(len(mx))]

                det_p = [prediction[j][*i].item() for j, i in enumerate(detected_coords)]
                detected_coords = [[i.item() for i in j] for j in detected_coords]

                # convert coords to original geometry

                # revert resize
                new_coordinates = convert_coordinates(detected_coords, data.shape[-3:], properties['shape_after_cropping_and_before_resampling'])
                # revert cropping
                crop_offset = [i[0] for i in properties['bbox_used_for_cropping']]
                new_coordinates = [[k + crop_offset[l] for l, k in enumerate(i)] for i in new_coordinates]

                # export coordinates
                save_json({i: {'coordinates': j, 'likelihood': l} for i, j, l in zip(range(1, 23), new_coordinates, det_p)}, join(validation_output_folder, k + '.json'))

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