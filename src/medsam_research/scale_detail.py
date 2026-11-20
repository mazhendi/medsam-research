"""Spatial scale selection and candidate-conditioned local detail retrieval.

No labels or hard masks enter these operators. Coordinates use pixel-center
alignment (align_corners=False); out-of-image neighbors are masked before softmax.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveScaleSelection(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.project = nn.Conv2d(channels, width, 1)
        self.branches = nn.ModuleList(nn.Sequential(
            nn.Conv2d(width, width, 3, padding=d, dilation=d, groups=width),
            nn.GELU()) for d in (1, 2, 3))
        self.controller = nn.Conv2d(3*width, 3, 1)
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        projected = self.project(x)
        z = [branch(projected) for branch in self.branches]
        weights = self.controller(torch.cat(z, 1)).float().softmax(1)
        mixed = sum(weights[:, i:i+1].to(v.dtype)*v for i, v in enumerate(z))
        return x+self.output(mixed), weights


class RegionDetailRetrieval(nn.Module):
    def __init__(self, semantic_channels, detail_channels, width, window=5, bias_strength=1.):
        super().__init__()
        self.window = window
        self.bias_strength = bias_strength
        self.query = nn.Conv2d(semantic_channels, width, 1)
        self.key = nn.Conv2d(detail_channels, width, 1)
        self.value = nn.Conv2d(detail_channels, width, 1)
        self.output = nn.Conv2d(width, semantic_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, semantic, detail, occupancy, candidate_weights):
        b, _, h, w = semantic.shape
        hd, wd = detail.shape[-2:]
        # Query centers map to the native detail grid; offsets remain native
        # detail pixels, so this is not downsampling D before neighborhood search.
        yy, xx = torch.meshgrid(torch.arange(h, device=detail.device),
                                torch.arange(w, device=detail.device), indexing='ij')
        radius = self.window//2
        offsets = torch.arange(-radius, radius+1, device=detail.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
        gx = (2*(xx.float()+.5)/w-1)[None] + 2*ox.flatten()[:, None, None]/wd
        gy = (2*(yy.float()+.5)/h-1)[None] + 2*oy.flatten()[:, None, None]/hd
        grid = torch.stack((gx.expand_as(gy), gy.expand_as(gx)), -1)
        n = self.window**2
        grid = grid.reshape(1, n*h, w, 2).expand(b, -1, -1, -1)

        def sample(tensor):
            # Explicit FP32 sampling avoids CPU bf16 incompatibility and keeps
            # candidate log affinities stable under CUDA autocast.
            sampled = F.grid_sample(tensor.float(), grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
            return sampled.reshape(b, tensor.shape[1], n, h, w)

        q = self.query(semantic).float()
        keys, values = sample(self.key(detail)), sample(self.value(detail))
        masks = F.interpolate(occupancy.float(), (hd, wd), mode='bilinear', align_corners=False)
        center = F.interpolate(occupancy.float(), (h, w), mode='bilinear', align_corners=False)
        neighbor = sample(masks)
        agreement = center[:, :, None]*neighbor+(1-center[:, :, None])*(1-neighbor)
        confidence = (2*center-1).abs()[:, :, None]*(2*neighbor-1).abs()
        # Uncertain candidates give neutral bias; confident disagreement is
        # penalized softly. Permuting candidates and weights leaves this invariant.
        affinity = (candidate_weights.float()[:, :, None, None, None]
                    * (.5+confidence*(agreement-.5))).sum(1)
        scores = (q[:, :, None]*keys).sum(1)/math.sqrt(q.shape[1])
        scores = scores+self.bias_strength*affinity.clamp_min(1e-6).log()
        coordinates = grid.reshape(b, n, h, w, 2)
        valid = ((coordinates[..., 0].abs() <= 1-1/wd+1e-6)
                 & (coordinates[..., 1].abs() <= 1-1/hd+1e-6))
        # The mapped center can fall between border pixels; keep center valid.
        valid[:, n//2] = True
        attention = scores.masked_fill(~valid, -torch.inf).softmax(1)
        retrieved = (attention[:, None]*values).sum(2)
        return semantic+self.output(retrieved.to(semantic.dtype)), attention
