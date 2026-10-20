"""Building blocks for the manuscript-based reference implementation."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return F.silu(x + self.body(x))


class LearnedResize(nn.Module):
    """Learned sampling offsets; independent implementation, not official DySample.

    Zero-initialized offsets give ordinary bilinear resizing with border padding.
    Offsets are expressed in source pixels and bounded to half a source pixel.
    """
    def __init__(self, channels):
        super().__init__()
        self.offset = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x, size):
        h, w = size
        delta = F.interpolate(self.offset(x), size=size, mode='bilinear', align_corners=False)
        delta = delta.tanh() * 0.5
        yy = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) * 2 / h - 1
        xx = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) * 2 / w - 1
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack((gx, gy), -1)[None].expand(x.shape[0], -1, -1, -1)
        scale = x.new_tensor([2 / x.shape[-1], 2 / x.shape[-2]])
        grid = grid + delta.permute(0, 2, 3, 1) * scale
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
