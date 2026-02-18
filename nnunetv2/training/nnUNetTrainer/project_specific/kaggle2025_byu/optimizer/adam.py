import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.losses.mse import MotorRegressionTrainer_MSELoss


class MotorRegressionTrainer_MSELoss_Adam3en4(MotorRegressionTrainer_MSELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                      amsgrad=False, fused=True, betas=(0.9, 0.999))
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class MotorRegressionTrainer_MSELoss_Adam3en3(MotorRegressionTrainer_MSELoss_Adam3en4):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-3


class MotorRegressionTrainer_MSELoss_Adam3en2(MotorRegressionTrainer_MSELoss_Adam3en4):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-2


class MotorRegressionTrainer_MSELoss_Adam3en5(MotorRegressionTrainer_MSELoss_Adam3en4):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-5

