"""HAMB-inspired progressive concatenation candidate; not a novelty claim."""
import torch
from torch import nn

from .selective_mamba import SelectiveMamba1


class ProgressiveConcatMambaB(nn.Module):
    """DW1/3/5/7 → concatenate previous output → PW → LN/Mamba.

    Four independent raster Mamba paths. No per-path output PW and no outer
    residual: these details follow the approved candidate diagram.
    """
    def __init__(self, width=32, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        if width <= 0 or width % 4:
            raise ValueError("width must be positive and divisible by four")
        group = width // 4
        self.project = nn.Conv2d(width, width, 1)
        self.local = nn.ModuleList(nn.Conv2d(group, group, k, padding=k//2, groups=group)
                                   for k in (1, 3, 5, 7))
        self.transition = nn.ModuleList(nn.Conv2d(2*group, group, 1) for _ in range(3))
        self.norm = nn.ModuleList(nn.LayerNorm(group) for _ in range(4))
        self.mamba = nn.ModuleList(SelectiveMamba1(group, d_state, d_conv, expand, backend)
                                   for _ in range(4))
        self.fuse = nn.Conv2d(width, width, 1)

    def forward(self, x):
        outputs = []
        for i, group in enumerate(self.project(x).chunk(4, dim=1)):
            feature = self.local[i](group)
            if i:
                feature = self.transition[i-1](torch.cat((feature, outputs[-1]), dim=1))
            batch, channels, height, width = feature.shape
            tokens = self.norm[i](feature.flatten(2).transpose(1, 2))
            outputs.append(self.mamba[i](tokens).transpose(1, 2).reshape(batch, channels, height, width))
        return self.fuse(torch.cat(outputs, dim=1))
