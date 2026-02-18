import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.regression import Nonlin_RegDice_loss, RegDice1
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.data_augmentation.more_DA import \
    MotorRegressionTrainer_BCEtopK20Loss_moreDA
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.losses.bce_topk import BCE_topK_loss
import numpy as np


class CompoundLoss(nn.Module):
    def __init__(self, *args, weights=None):
        self.losses = args
        if weights is None:
            self.weights = [1] * len(args)
        else:
            self.weights = weights
            assert len(weights) == len(args)
        super().__init__()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        loss = None
        for i, j in zip(self.losses, self.weights):
            if loss is None:
                loss = i(net_output, target) * j
            else:
                loss += i(net_output, target) * j
        return loss


class MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def _build_loss(self):
        loss = CompoundLoss(BCE_topK_loss(k=20), Nonlin_RegDice_loss(RegDice1(), F.sigmoid), weights=(0.5, 0.5))

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.soft_dice)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not sself._do_i_compile():
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


class Scheduled_MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA(MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA):
    def on_train_epoch_start(self):
        weights = (0.5 + self.current_epoch / self.num_epochs * 0.5,
                                     max(0, 0.5 - self.current_epoch / self.num_epochs * 0.5)
                                     )
        print(weights)
        loss = CompoundLoss(BCE_topK_loss(k=20), Nonlin_RegDice_loss(RegDice1(), F.sigmoid),
                            weights=weights)

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
        self.loss = loss
        return super().on_train_epoch_start()