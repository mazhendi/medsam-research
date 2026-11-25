"""B1 candidate: three local scales, cross-scale interaction, then Mamba-1."""
import math

import torch
from torch import nn

from .selective_mamba import SelectiveMamba1


class CrossScaleInteraction(nn.Module):
    """Per-pixel attention to the other two scales, preserving own features.

    There is no spatial attention matrix or residual addition. Q/K/V weights
    are independent per scale; a source's K/V are reused by the other targets.
    """
    def __init__(self, width=16):
        super().__init__()
        self.query = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(3))
        self.key = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(3))
        self.value = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(3))
        self.fuse = nn.ModuleList(nn.Conv2d(2 * width, width, 1) for _ in range(3))
        self.scale = 1 / math.sqrt(width)

    def forward(self, features, return_weights=False):
        if len(features) != 3:
            raise ValueError("B1 interaction requires three scale features")
        queries = [layer(x) for layer, x in zip(self.query, features)]
        keys = [layer(x) for layer, x in zip(self.key, features)]
        values = [layer(x) for layer, x in zip(self.value, features)]
        outputs, attention = [], []
        for i in range(3):
            other = [j for j in range(3) if j != i]
            # Softmax is only over the two other scales, at the same pixel.
            dtype = torch.float32 if queries[i].dtype in (torch.float16, torch.bfloat16) else queries[i].dtype
            scores = torch.stack([
                (queries[i].to(dtype) * keys[j].to(dtype)).sum(1) * self.scale
                for j in other
            ], dim=1)
            weights = scores.softmax(1).to(values[i].dtype)
            message = (weights[:, 0:1] * values[other[0]]
                       + weights[:, 1:2] * values[other[1]])
            outputs.append(self.fuse[i](torch.cat((features[i], message), dim=1)))
            attention.append(weights)
        return (outputs, attention) if return_weights else outputs


class CrossScaleMambaB1(nn.Module):
    """32→48, independent DW3/5/7, scale interaction, three Mamba paths, 48→32.

    No DW9, shuffle, inter-group addition or outer residual. Mamba's standard
    internal state recurrence/skip D is retained. Input/output grids unchanged.
    """
    def __init__(self, width=32, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        group = 16
        self.project = nn.Conv2d(width, 3 * group, 1)
        self.local = nn.ModuleList(nn.Conv2d(group, group, k, padding=k // 2, groups=group)
                                   for k in (3, 5, 7))
        self.interaction = CrossScaleInteraction(group)
        self.norm = nn.ModuleList(nn.LayerNorm(group) for _ in range(3))
        self.mamba = nn.ModuleList(SelectiveMamba1(group, d_state, d_conv, expand, backend)
                                   for _ in range(3))
        self.mix = nn.ModuleList(nn.Conv2d(group, group, 1) for _ in range(3))
        self.fuse = nn.Conv2d(3 * group, width, 1)

    def forward(self, x):
        groups = self.project(x).chunk(3, dim=1)
        local = [layer(group) for layer, group in zip(self.local, groups)]
        interacted = self.interaction(local)
        outputs = []
        for feature, norm, mamba, mix in zip(interacted, self.norm, self.mamba, self.mix):
            batch, channels, height, width = feature.shape
            tokens = norm(feature.flatten(2).transpose(1, 2))
            mixed = mamba(tokens).transpose(1, 2).reshape(batch, channels, height, width)
            outputs.append(mix(mixed))
        return self.fuse(torch.cat(outputs, dim=1))
