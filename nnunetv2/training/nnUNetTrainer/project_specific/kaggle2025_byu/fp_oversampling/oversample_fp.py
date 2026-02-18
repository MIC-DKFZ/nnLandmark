from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Union, Tuple, List

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from torch._dynamo import OptimizedModule

from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.data_augmentation.more_DA import \
    MotorRegressionTrainer_BCEtopK20Loss_moreDA
from nnunetv2.utilities.helpers import empty_cache


class CustomRemoveLabelTansform(SegOnlyTransform):
    def __init__(self, label_values: List[int], set_to: int, segmentation_channels: Union[int, Tuple[int, ...], List[int]] = None):
        if not isinstance(segmentation_channels, (list, tuple)) and segmentation_channels is not None:
            segmentation_channels = [segmentation_channels]
        self.segmentation_channels = segmentation_channels
        self.label_values = torch.Tensor(label_values)
        self.set_to = set_to
        super().__init__()

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if self.segmentation_channels is None:
            channels = list(range(segmentation.shape[0]))
        else:
            channels = self.segmentation_channels
        for s in channels:
            # mask =
            # if torch.any(mask):
            #     deleted = torch.unique(segmentation[s][mask])
            #     print('deleted', deleted)
            segmentation[s][torch.isin(segmentation[s], self.label_values)] = self.set_to
            # del mask
        return segmentation


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
             device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.probabilistic_oversampling = True
        self.oversample_foreground_percent = 0.4

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
        transforms: ComposeTransforms = super().get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales,
                                                     mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm,
                                                     is_cascaded, foreground_labels, regions, ignore_label)
        transforms.transforms.insert(-3,
                                     CustomRemoveLabelTansform(list(range(20, 30)), set_to=0))
        print(transforms)
        return transforms

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> BasicTransform:
        transforms: ComposeTransforms = super().get_validation_transforms(deep_supervision_scales, is_cascaded,
                                                                          foreground_labels, regions, ignore_label)
        transforms.transforms.insert(-3,
                                     CustomRemoveLabelTansform(list(range(20, 30)), set_to=0))
        print(transforms)
        return transforms


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling_3kep(MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
             device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 3000


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_3kep(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
             device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 3000


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_5kep(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5000


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_5kep_EDT20(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5000
        self.min_motor_distance = 20


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_5kep_EDT25(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5000
        self.min_motor_distance = 25

class MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT20(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 3500
        self.min_motor_distance = 20


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 3500
        self.min_motor_distance = 25

class MotorRegressionTrainer_BCEtopK20Loss_moreDA_EDT25(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.min_motor_distance = 25

class MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 3500


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling_warmup30_train1500_initlr1en3(MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500
        self.warmup_duration_whole_net = 30  # lin increase whole network
        self.initial_lr = 1e-3
        self.training_stage = None  # 'warmup_all', 'train'

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.SGD(
                params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        We need to overwrite that entire function because we need to fiddle the correct optimizer in between
        loading the checkpoint and applying the optimizer states. Yuck.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else self.inference_allowed_mirroring_axes
        )

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        if self.current_epoch < self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_warmup50_train2000_initlr1en3(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.initial_lr = 1e-3
        self.training_stage = None  # 'warmup_all', 'train'

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.SGD(
                params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        We need to overwrite that entire function because we need to fiddle the correct optimizer in between
        loading the checkpoint and applying the optimizer states. Yuck.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else self.inference_allowed_mirroring_axes
        )

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        if self.current_epoch < self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_1en1(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-1

class MotorRegressionTrainer_BCEtopK20Loss_moreDA_1en3(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3