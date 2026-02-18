import torch

from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.losses.mse import MotorRegressionTrainer_MSELoss


class MotorRegressionTrainer_MSELoss_lr1en3(MotorRegressionTrainer_MSELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3


class MotorRegressionTrainer_MSELoss_lr1en1(MotorRegressionTrainer_MSELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-1


class MotorRegressionTrainer_MSELoss_lr1en4(MotorRegressionTrainer_MSELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4


