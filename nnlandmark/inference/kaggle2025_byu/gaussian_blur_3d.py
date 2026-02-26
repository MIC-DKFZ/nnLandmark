import torch
from torch.nn import functional as F

class GaussianBlur3D:
    def __init__(self, sigma: float, truncate: float = 4.0, device=None):
        self.sigma = sigma
        self.truncate = truncate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel_z, self.kernel_y, self.kernel_x = self._get_separable_gaussian_kernels_3d(sigma, truncate)
        self.kernel_z = self.kernel_z.to(self.device)
        self.kernel_y = self.kernel_y.to(self.device)
        self.kernel_x = self.kernel_x.to(self.device)

    @staticmethod
    def _gaussian_1d_kernel(sigma: float, truncate: float) -> torch.Tensor:
        radius = int(truncate * sigma + 0.5)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel.view(1, 1, -1)  # Shape: [1, 1, k]

    def _get_separable_gaussian_kernels_3d(self, sigma: float, truncate: float):
        kernel_1d = self._gaussian_1d_kernel(sigma, truncate)  # [1, 1, k]
        k = kernel_1d.shape[-1]

        kernel_z = kernel_1d.view(1, 1, k, 1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, k, 1)
        kernel_x = kernel_1d.view(1, 1, 1, 1, k)

        return kernel_z, kernel_y, kernel_x

    def apply(self, volume: torch.Tensor) -> torch.Tensor:
        # Padding sizes
        pad_z = self.kernel_z.shape[2] // 2
        pad_y = self.kernel_y.shape[3] // 2
        pad_x = self.kernel_x.shape[4] // 2

        # Pad the input tensor: (W, H, D) order
        volume = F.pad(volume, (pad_x, pad_x, pad_y, pad_y, pad_z, pad_z), mode='replicate')

        # Apply depthwise separable convolutions
        volume = F.conv3d(volume, self.kernel_z, groups=volume.shape[1], padding=0)
        volume = F.conv3d(volume, self.kernel_y, groups=volume.shape[1], padding=0)
        volume = F.conv3d(volume, self.kernel_x, groups=volume.shape[1], padding=0)

        return volume