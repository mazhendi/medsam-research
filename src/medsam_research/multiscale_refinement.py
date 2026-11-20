"""Coarse-to-fine refinement with native query grids and local coarse K/V."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GatedFFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = 2*channels
        self.norm = ChannelNorm(channels)
        self.expand = nn.Conv2d(channels, 2*hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.output = nn.Conv2d(hidden, channels, 1)
        zero_output(self.output)

    def forward(self, x):
        a, b = self.expand(self.norm(x)).chunk(2, 1)
        return x+self.output(F.gelu(self.spatial(a))*b)


class CoarseGuidedAttention(nn.Module):
    def __init__(self, fine_channels, coarse_channels, width, window):
        super().__init__()
        self.window = window
        self.fine_norm = ChannelNorm(fine_channels)
        self.coarse_norm = ChannelNorm(coarse_channels)
        self.query = nn.Conv2d(fine_channels, width, 1)
        self.key = nn.Conv2d(coarse_channels, width, 1)
        self.value = nn.Conv2d(coarse_channels, width, 1)
        self.output = nn.Conv2d(width, fine_channels, 1)
        zero_output(self.output)
        self.ffn = GatedFFN(fine_channels)

    def forward(self, fine, coarse):
        b, _, h, w = fine.shape
        hc, wc = coarse.shape[-2:]
        q = self.query(self.fine_norm(fine)).float()
        normalized = self.coarse_norm(coarse)
        key, value = self.key(normalized).float(), self.value(normalized).float()
        yy, xx = torch.meshgrid(torch.arange(h, device=fine.device),
                                torch.arange(w, device=fine.device), indexing='ij')
        radius = self.window//2
        axis = torch.arange(-radius, radius+1, device=fine.device)
        oy, ox = torch.meshgrid(axis, axis, indexing='ij')
        gx = (2*(xx.float()+.5)/w-1)[None]+2*ox.flatten()[:, None, None]/wc
        gy = (2*(yy.float()+.5)/h-1)[None]+2*oy.flatten()[:, None, None]/hc
        grid = torch.stack((gx.expand_as(gy), gy.expand_as(gx)), -1)
        n = self.window**2
        valid = ((grid[..., 0].abs() <= 1+1e-6) & (grid[..., 1].abs() <= 1+1e-6))
        grid = grid.reshape(1, n*h, w, 2).expand(b, -1, -1, -1)

        def sample(tensor):
            # Half-pixel mapping, native coarse-pixel offsets. Border extension
            # supports fine query centers near the outermost coarse centers.
            return F.grid_sample(tensor, grid, padding_mode='border',
                                 align_corners=False).reshape(b, q.shape[1], n, h, w)

        keys, values = sample(key), sample(value)
        score = (q[:, :, None]*keys).sum(1)/math.sqrt(q.shape[1])
        weights = score.masked_fill(~valid[None], -torch.inf).softmax(1)
        retrieved = (weights[:, None]*values).sum(2)
        updated = fine+self.output(retrieved.to(fine.dtype))
        return self.ffn(updated)


class MultiScaleRefinement(nn.Module):
    def __init__(self, channels, width=32, window=3):
        super().__init__()
        self.coarse = GatedFFN(channels[2])
        self.middle = CoarseGuidedAttention(channels[1], channels[2], width, window)
        self.fine = CoarseGuidedAttention(channels[0], channels[1], width, window)

    def forward(self, pyramid):
        f1, f2, f3 = [x.float() for x in pyramid]
        f3 = self.coarse(f3)
        f2 = self.middle(f2, f3)
        f1 = self.fine(f1, f2)
        return [f1, f2, f3]
