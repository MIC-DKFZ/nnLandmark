import multiprocessing
from copy import deepcopy
from time import time, sleep
from typing import Union, Tuple, List

import SimpleITK
import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, load_json
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from torch import distributed as dist, autocast
from torch import nn
from torch.nn import functional as F

from nnunetv2.configuration import default_num_processes
from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates
from nnunetv2.evaluation.kaggle2025_byu import compute_f_beta
from nnunetv2.inference.kaggle2025_byu.gaussian_blur_3d import GaussianBlur3D
from nnunetv2.inference.kaggle2025_byu.iterative_maxpool import iterative_3x3_same_padding_pool3d
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import ConvertSegToRegrTarget
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.regression import RegDice_and_MSE_loss, RegDice1, RegDice3
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import dummy_context


class MotorRegressionTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 50
        self.min_motor_distance: int = 15

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
        transforms.transforms.append(ConvertSegToRegrTarget('EDT', gaussian_sigma=self.min_motor_distance // 3,
                                                            edt_radius=self.min_motor_distance))
        transforms.transforms.append(DownsampleSegForDSTransform(deep_supervision_scales))

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
        assert isinstance(transforms.transforms[-1], DownsampleSegForDSTransform)
        transforms.transforms.pop(-1)


        transforms.transforms.append(ConvertSegToRegrTarget('EDT', gaussian_sigma=self.min_motor_distance // 3,
                                                            edt_radius=self.min_motor_distance))
        transforms.transforms.append(DownsampleSegForDSTransform(deep_supervision_scales))

        return transforms

    def _build_loss(self):
        # loss = Nonlin_MSE_loss()
        loss = RegDice_and_MSE_loss(regdice=RegDice1())

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.soft_dice)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

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
        l = self.loss([i.float() for i in output], target)

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

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling,
                                 random_offset=[i // 3 for i in self.configuration_manager.patch_size])
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
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

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_output_channels = 1
        net = nnUNetTrainer.build_network_architecture(architecture_class_name, arch_init_kwargs,
                                                       arch_init_kwargs_req_import, num_input_channels,
                                                       num_output_channels, enable_deep_supervision)
        return net

    @torch.inference_mode()
    def perform_actual_validation(self, save_probabilities: bool = False):
        MIN_P = 0.1 # detection threshold. Designed to export a lot of motors, must be calibrated on 5 fold cv

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
        dsj['labels'] = {'background': 0}
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        dsj, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        gb = GaussianBlur3D(2, 3, self.device)
        orig_shapes = load_json(join(nnUNet_raw, self.plans_manager.dataset_name, 'train_OrigShapes.json'))

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

                # print('pred')
                prediction = predictor.predict_sliding_window_return_logits(data)
                # print('sigmoid')
                prediction = F.sigmoid(prediction).float()[0]

                # convert prediction to motor localizations
                # smooth with Gaussian filter (sigma 2)
                # print('smooth')
                smooth_pred = gb.apply(prediction[None, None])[0, 0]

                # extract local maxima along with their motor probabilities
                # we hackily reduce the max pool range because the max pooling kernel will be a square, not a circle. This is super random but eh.
                # print('maxpool')
                mp = iterative_3x3_same_padding_pool3d(smooth_pred[None, None], self.min_motor_distance - 2)[0, 0]
                # print('detect')
                detections = (smooth_pred == mp) & (prediction > MIN_P)
                detected_coords = torch.argwhere(detections)
                # print('done')

                det_p = [prediction[*i].item() for i in detected_coords]
                detected_coords = [[i.item() for i in j] for j in detected_coords]

                # convert coords to original
                coords_in_orig_shape = convert_coordinates(detected_coords, prediction.shape, orig_shapes[k])

                # save raw prediction, found motors with their probs, motor_map

                # raw prediction
                # if os.environ.get('nnUNet_save_soft_preds'):
                tmp = (torch.round(prediction * 100) / 100).cpu().numpy()
                tmp = SimpleITK.GetImageFromArray(tmp)
                tmp.SetSpacing(list(properties['spacing'])[::-1])
                SimpleITK.WriteImage(tmp, output_filename_truncated + '.nii.gz')
                del tmp

                # detection map
                # if os.environ.get('nnUNet_save_hard_preds'):
                #     tmp = SimpleITK.GetImageFromArray(detections.cpu().numpy().astype(np.uint8))
                #     tmp.SetSpacing(list(properties['spacing'])[::-1])
                #     SimpleITK.WriteImage(tmp, output_filename_truncated + '_detections.nii.gz')
                #     from skimage.morphology.gray import dilation
                #     prediction_exp = torch.clone(prediction)
                #     prediction_exp[~detections] = 0
                #     prediction_dilated = dilation(prediction_exp.cpu().numpy(), footprint=ball(4))
                #     tmp = SimpleITK.GetImageFromArray(prediction_dilated)
                #     tmp.SetSpacing(list(properties['spacing'])[::-1])
                #     SimpleITK.WriteImage(tmp, output_filename_truncated + '_dil.nii.gz')
                #     del prediction_exp

                # coordinates and probabilities
                save_json({'coordinates': detected_coords, 'coordinates_orig_shape': coords_in_orig_shape, 'probabilities': det_p}, output_filename_truncated + '.json')

        gt = load_json(join(nnUNet_raw, self.plans_manager.dataset_name, 'train_coordinates.json'))
        gt_list = [np.array(gt[k]) for k in val_keys]
        preds = [load_json(join(validation_output_folder, k + '.json')) for k in val_keys]

        probs = np.linspace(MIN_P, 1, num=200) #np.unique([j for i in preds for j in i['probabilities']])
        fbeta = []
        for threshold in probs:
            pred_list_here = []
            for i in range(len(preds)):
                pred_list_here.append(np.array([preds[i]['coordinates'][j] for j in range(len(preds[i]['probabilities'])) if preds[i]['probabilities'][j] > threshold]))
            # we need to threshold the prediction
            fbeta.append(compute_f_beta(pred_list_here, gt_list, 2, 35))

        import seaborn as sns
        # Plot using seaborn
        sns.set(style="whitegrid", context="talk")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=probs, y=fbeta, marker='o')
        plt.xlabel("Probability Threshold", fontsize=14)
        plt.ylabel("F\u03B2 Score (β=2)", fontsize=14)
        plt.title("F\u03B2 Score vs Probability Threshold", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(join(validation_output_folder, 'f2_vs_threshold.png'))

        fbeta_2 = []
        probs2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in probs2:
            pred_list_here = []
            for i in range(len(preds)):
                pred_list_here.append(np.array([preds[i]['coordinates'][j] for j in range(len(preds[i]['probabilities'])) if preds[i]['probabilities'][j] > threshold]))
            # we need to threshold the prediction
            fbeta_2.append(compute_f_beta(pred_list_here, gt_list, 2, 35))

        save_json(
            {
                'f2_max': np.max(fbeta),
                'f2_at_0.5': fbeta_2[4],
                'rough_sweep': {i: j for i, j in zip(probs2, fbeta_2)},
            }, join(validation_output_folder, 'scores.json')
        )
        save_json(
            {
                'precise_sweep': {i: j for i, j in zip(probs, fbeta)},
            }, join(validation_output_folder, 'scores_precise.json')
        )

    def on_epoch_end(self):
        ### ema logs val loss here, lower is better
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=6))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=6))
        # self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
        #                                        self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # logs val loss instead of fg dice
        outputs_collated = collate_outputs(val_outputs)

        if self.is_ddp:
            world_size = dist.get_world_size()

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        # global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        # mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', loss_here, self.current_epoch)
        self.logger.log('dice_per_class_or_region', [1, 1], self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)
        # import IPython;IPython.embed()
        # if False:
        # idx = 1
        # from batchviewer import view_batch
        # tmp = F.sigmoid(output[0][idx, 0])
        # print(tmp.max())
        # view_batch(data[idx, 0], target[0][idx, 0], tmp)
        # quit()

        return {'loss': l.detach().cpu().numpy()}


class MotorRegressionTrainer_MSERegDice3(MotorRegressionTrainer):
    def _build_loss(self):
        loss = RegDice_and_MSE_loss(regdice=RegDice3())

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.soft_dice)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
