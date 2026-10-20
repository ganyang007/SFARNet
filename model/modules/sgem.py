"""SGEM: structure guidance, coarse context and SimAM-style importance."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import LearnedResize


class SGEM(nn.Module):
    """[B,C,H,W] -> [B,out_channels,H,W]; default output has 2C channels."""
    def __init__(self, channels, out_channels=None, pool_factor=2, energy_lambda=1e-4):
        super().__init__()
        if channels < 1 or pool_factor < 1 or energy_lambda <= 0:
            raise ValueError('channels/pool_factor/lambda must be positive')
        out_channels = 2 * channels if out_channels is None else out_channels
        self.pool_factor, self.energy_lambda = pool_factor, energy_lambda
        self.structure = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.structure_gate = nn.Conv2d(1, channels, 3, padding=1)
        self.project = nn.Conv2d(channels, channels, 1)
        self.context = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                     nn.Conv2d(channels, channels, 1), nn.SiLU())
        self.resize = LearnedResize(channels)
        self.select = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.SiLU(),
                                    nn.Conv2d(channels, channels, 1))
        self.out1 = nn.Conv2d(channels, out_channels, 1)
        self.out3 = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x):
        structure = self.structure(x).mean(1, keepdim=True)
        structure = F.layer_norm(structure, structure.shape[-2:])
        gate = self.structure_gate(structure).sigmoid()
        u = self.project(x)
        size = (max(1, u.shape[-2] // self.pool_factor), max(1, u.shape[-1] // self.pool_factor))
        context = self.resize(self.context(F.adaptive_avg_pool2d(u, size)), u.shape[-2:])
        # Standard efficient channel-statistics approximation to SimAM energy.
        deviation = (u - u.mean((-2, -1), keepdim=True)).square()
        variance = deviation.sum((-2, -1), keepdim=True) / max(u.shape[-2] * u.shape[-1] - 1, 1)
        importance = (deviation / (4 * (variance + self.energy_lambda)) + 0.5).sigmoid()
        enhanced = u * (context + importance)
        fused = enhanced * (1 + gate)
        selected = self.select(fused)
        weights = (selected.mean((-2, -1), keepdim=True) + selected.amax((-2, -1), keepdim=True)).sigmoid()
        refined = x + fused * weights
        return self.out1(refined) + self.out3(refined)
