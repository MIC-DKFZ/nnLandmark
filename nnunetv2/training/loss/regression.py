import torch
from torch.nn import functional as F, MSELoss
from torch import nn

class RegDice1(nn.Module):
    def __init__(self):
        """
        Original implementation by fabi
        """
        super(RegDice1, self).__init__()

        self.eps = 1e-6

    def forward(self, x, y):
        """
        x must be (b, c, x, y, z), y must be same shape as x
        x and y must be torch tensors
        x is network output

        THIS IS THE ONE

        :param x:
        :param y:
        :return:
        """
        assert all([i == j for i, j in zip(x.shape, y.shape)])

        axes = list(range(2, len(x.shape)))

        tp = (F.relu(y - torch.abs(x - y), inplace=True)).sum(axes)
        fp = (F.relu(x - y, inplace=True)).sum(axes)
        fn = (F.relu(y - x, inplace=True)).sum(axes)

        return (- (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)).mean()


class RegDice2(nn.Module):
    def __init__(self):
        """
        Awesome improvements by Fabian:
        FP cannot exceed gt
        TP cannot flatline to 0
        symmetric loss (shape is the same/mirrored between gt=0.3 and 0.7)
        """
        super(RegDice2, self).__init__()

        self.eps = 1e-6

    def forward(self, x, y):
        assert all([i == j for i, j in zip(x.shape, y.shape)])
        axes = list(range(2, len(x.shape)))

        tp = torch.where(x < y, x, y - y * (x - y) / (1 - y)).sum(axes)
        fp = F.relu((x - y) / (1 - y) * y, inplace=True).sum(axes)
        fn = F.relu(y - x, inplace=True).sum(axes)

        return (- (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)).mean()


class RegDice3(nn.Module):
    def __init__(self):
        """
        Kaggle implementation
        """
        super(RegDice3, self).__init__()

    def forward(self, x, y):
        assert all([i == j for i, j in zip(x.shape, y.shape)])
        axes = list(range(2, len(x.shape)))

        intersection = torch.sum((1 - torch.abs(y - x)) * torch.minimum(y, x), dim=axes)  # CHANGED FROM ORIGINAL
        cardinality = torch.sum(x + y, dim=axes)
        return (-(2.0 * intersection) / cardinality.clamp_min(1e-6)).mean()


class RegDice_and_MSE_loss(nn.Module):
    def __init__(self, regdice, nonlinearity=F.sigmoid):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.soft_dice = regdice
        self.mse = MSELoss(reduction='mean')

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        net_output = self.nonlinearity(net_output)
        return self.soft_dice(net_output, target) + self.mse(net_output, target)

class Nonlin_MSE_loss(nn.Module):
    def __init__(self, nonlinearity=F.sigmoid):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.mse = MSELoss(reduction='mean')

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        net_output = self.nonlinearity(net_output)
        return self.mse(net_output, target)

class Nonlin_RegDice_loss(nn.Module):
    def __init__(self, regdice, nonlinearity=F.sigmoid):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.regdice = regdice

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        net_output = self.nonlinearity(net_output)
        return self.regdice(net_output, target)