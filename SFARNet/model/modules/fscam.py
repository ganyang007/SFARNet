"""Spatial/frequency enhancement and differentiable mini-batch linear TTT."""
import torch
from torch import nn
from torch.nn import functional as F


class FSCE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.base = nn.Conv2d(channels, channels, 1)
        self.gradient = nn.Conv2d(2 * channels, channels, 1)
        kernel = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]) / 32
        self.register_buffer('kx', kernel[None, None].repeat(channels, 1, 1, 1))
        self.register_buffer('ky', kernel.T[None, None].repeat(channels, 1, 1, 1))
        self.spectral = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.attention = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        u = self.base(x)
        padded = F.pad(u, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(padded, self.kx.to(u.dtype), groups=self.channels)
        gy = F.conv2d(padded, self.ky.to(u.dtype), groups=self.channels)
        spatial = u + self.gradient(torch.cat((gx, gy), 1))
        # Float32 FFT avoids half-precision FFT size restrictions.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = torch.fft.rfft2(spatial.float(), norm='ortho')
            packed = torch.cat((z.real, z.imag), 1)
            packed = F.conv2d(packed, self.spectral.weight.float(),
                              self.spectral.bias.float())
            real, imag = packed.chunk(2, 1)
            freq = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[-2:], norm='ortho')
        freq = freq.to(spatial.dtype)
        stats = torch.cat((freq.mean(1, keepdim=True), freq.amax(1, keepdim=True)), 1)
        return freq * self.attention(stats).sigmoid()


class LocalityAwareAdaptation(nn.Module):
    """Nonpersistent per-sample linear state, updated in chunks of eight tokens.

    Implements the exact analytic gradient of ||K W - V||² / (2 * tokens).
    No autograd.grad is required at inference; gradients through updates remain
    available in training. This is a readable sequential reference, not a fused
    or speed-optimized TTT kernel. hidden_dim defaults to C as an explicit
    reconstruction assumption, not a recovered manuscript formula.
    """
    def __init__(self, channels, hidden_dim=None, chunk_size=8, learning_rate=0.001):
        super().__init__()
        d = channels if hidden_dim is None else hidden_dim
        if d < 1 or chunk_size < 1 or learning_rate <= 0:
            raise ValueError('hidden_dim/chunk_size/learning_rate must be positive')
        self.chunk_size, self.learning_rate = chunk_size, learning_rate
        self.project = nn.Conv2d(channels, d, 1)
        self.key = nn.Sequential(nn.Conv2d(d, d, 1), nn.Conv2d(d, d, 3, padding=1, groups=d))
        self.query = nn.Sequential(nn.Conv2d(d, d, 1), nn.Conv2d(d, d, 3, padding=1, groups=d))
        self.value = nn.Conv2d(d, d, 1)
        self.gate = nn.Conv2d(d, d, 1)
        self.initial_state = nn.Parameter(torch.eye(d))
        self.norm = nn.LayerNorm(d)
        self.output = nn.Conv2d(d, channels, 1)

    @staticmethod
    def sequence(x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, x):
        z = self.project(x)
        k, q, v = [self.sequence(branch(z)) for branch in (self.key, self.query, self.value)]
        # State calculations stay float32; no mutation of learned parameters.
        with torch.autocast(device_type=x.device.type, enabled=False):
            k, q, v = k.float(), q.float(), v.float()
            state = self.initial_state.float()[None].expand(x.shape[0], -1, -1)
            outputs = []
            for start in range(0, k.shape[1], self.chunk_size):
                kb = k[:, start:start + self.chunk_size]
                vb = v[:, start:start + self.chunk_size]
                error = kb @ state - vb
                gradient = kb.transpose(1, 2) @ error / kb.shape[1]
                state = state - self.learning_rate * gradient
                outputs.append(q[:, start:start + self.chunk_size] @ state)
            y = torch.cat(outputs, 1)
        y = self.norm(y.to(z.dtype)) * F.silu(self.sequence(self.gate(z)))
        y = y.transpose(1, 2).reshape_as(z)
        return self.output(y)


class FSCAM(nn.Module):
    """[B,C,H,W] -> [B,C,H,W]."""
    def __init__(self, channels, hidden_dim=None, chunk_size=8, learning_rate=0.001):
        super().__init__()
        self.enhance = FSCE(channels)
        self.adapt = LocalityAwareAdaptation(channels, hidden_dim, chunk_size, learning_rate)

    def forward(self, x):
        return self.adapt(self.enhance(x))
