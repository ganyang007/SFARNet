"""SFARNet reference network. See docs/IMPLEMENTATION_NOTES.md for assumptions."""
import torch
from torch import nn
from torch.nn import functional as F
from model.modules import SGEM, FSCAM, CAWR
from model.modules.common import ResidualBlock


class SFARNet(nn.Module):
    """Five-scale encoder/decoder with six probability maps for supervision.

    Default widths: 16,32,64,128,256. Forward returns M0 (probabilities).
    Set return_aux=True to return [M0,M1,M2,M3,M4,fused], all at input size.
    Inputs are padded to a multiple of 16 and predictions are cropped back.
    """
    def __init__(self, base_channels=16):
        super().__init__()
        if base_channels < 1:
            raise ValueError('base_channels must be positive')
        widths = [base_channels * 2**i for i in range(5)]
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 1), ResidualBlock(widths[0]))
        self.encoder = nn.ModuleList([SGEM(c) for c in widths[:-1]])
        self.skip = nn.ModuleList([FSCAM(c) for c in widths[:-1]])
        self.decoder = nn.ModuleList([CAWR(c) for c in widths[:-1]])
        self.heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in widths])
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('Expected grayscale BCHW input with one channel')
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode='replicate')
        features = [self.stem(x)]
        for block in self.encoder:
            features.append(F.avg_pool2d(block(features[-1]), 2))
        decoded = [None] * 4 + [features[-1]]
        for i in reversed(range(4)):
            decoded[i] = self.decoder[i](self.skip[i](features[i]), decoded[i+1])
        m0 = self.heads[0](decoded[0]).sigmoid()
        if not return_aux:
            return m0[..., :h, :w]
        maps = [m0] + [F.interpolate(self.heads[i](decoded[i]).sigmoid(),
                                    size=x.shape[-2:], mode='bilinear', align_corners=False)
                       for i in range(1, 5)]
        maps.append(self.fuse(torch.cat(maps, 1)).sigmoid())
        return [m[..., :h, :w] for m in maps]
