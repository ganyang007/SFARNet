"""Haar coefficient fusion with deep-conditioned channel modulation."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import ResidualBlock


def haar_dwt(x):
    """Orthonormal Haar transform: band order LL, LH, HL, HH.

    Odd sizes use replicated right/bottom padding; save size for inverse crop.
    """
    h, w = x.shape[-2:]
    x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.cat(((a+b+c+d)/2, (-a-b+c+d)/2,
                      (-a+b-c+d)/2, (a-b-c+d)/2), 1)


def haar_idwt(coefficients, size=None):
    if coefficients.shape[1] % 4:
        raise ValueError('Haar coefficient channels must be divisible by four')
    ll, lh, hl, hh = coefficients.chunk(4, 1)
    a, b = (ll-lh-hl+hh)/2, (ll-lh+hl-hh)/2
    c, d = (ll+lh-hl-hh)/2, (ll+lh+hl+hh)/2
    top = torch.stack((a, b), -1).flatten(-2)
    bottom = torch.stack((c, d), -1).flatten(-2)
    result = torch.stack((top, bottom), -2).flatten(-3, -2)
    return result if size is None else result[..., :size[0], :size[1]]


class CAWR(nn.Module):
    """shallow [B,C,H,W], deep [B,2C,ceil(H/2),ceil(W/2)] -> [B,C,H,W]."""
    def __init__(self, channels, deep_channels=None):
        super().__init__()
        deep_channels = 2 * channels if deep_channels is None else deep_channels
        if deep_channels % channels:
            raise ValueError('deep_channels must be divisible by shallow channels')
        self.channels = channels
        self.deep_project = nn.Conv2d(deep_channels, 4 * channels, 1, groups=channels)
        self.channel_weights = nn.Linear(4 * channels, channels)
        self.shallow_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.refine = ResidualBlock(channels)

    def forward(self, shallow, deep):
        bands = haar_dwt(shallow)
        if deep.shape[0] != shallow.shape[0] or deep.shape[-2:] != bands.shape[-2:]:
            raise ValueError('deep batch/size must match the shallow Haar coefficient grid')
        b, _, h, w = bands.shape
        # Group convolution produces channel-major groups; convert to band-major.
        projected = self.deep_project(deep).reshape(b, self.channels, 4, h, w)
        projected = projected.transpose(1, 2).reshape(b, 4 * self.channels, h, w)
        weight = self.channel_weights(projected.mean((-2, -1))).sigmoid()[:, None, :, None, None]
        fused = (bands.reshape(b, 4, self.channels, h, w) * self.shallow_scale[:, None]
                 + projected.reshape(b, 4, self.channels, h, w)) * weight
        reconstructed = haar_idwt(fused.reshape_as(bands), shallow.shape[-2:])
        return self.refine(reconstructed)
