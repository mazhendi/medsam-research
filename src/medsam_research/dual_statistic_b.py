"""Dual-statistic multiscale Mamba candidate, matching the approved diagram."""
import torch
from torch import nn

from .selective_mamba import SelectiveMamba1


class StatisticMambaPath(nn.Module):
    """Stride-one pooling → parallel DW3/5/7 → projection → LN → Mamba."""
    def __init__(self, width, statistic, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        if statistic == "average":
            # Exclude padding from averages to avoid artificial border attenuation.
            self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        elif statistic == "maximum":
            self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        else:
            raise ValueError("statistic must be average or maximum")
        self.local = nn.ModuleList(nn.Conv2d(width, width, k, padding=k // 2, groups=width)
                                   for k in (3, 5, 7))
        self.project = nn.Conv2d(3 * width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.mamba = SelectiveMamba1(width, d_state, d_conv, expand, backend)

    def forward(self, p):
        batch, channels, height, width = p.shape
        pooled = self.pool(p)
        feature = self.project(torch.cat([layer(pooled) for layer in self.local], dim=1))
        tokens = self.norm(feature.flatten(2).transpose(1, 2))
        return self.mamba(tokens).transpose(1, 2).reshape(batch, channels, height, width)


class DualStatisticMambaB(nn.Module):
    """Independent average/maximum paths, concatenation fusion, optional X skip.

    Both paths keep the full channel count and grid. No channel grouping,
    cross-attention, parameter sharing, or custom SSM parameterization.
    """
    def __init__(self, width=32, d_state=16, d_conv=4, expand=2, backend="auto", residual=True):
        super().__init__()
        self.residual = residual
        self.project = nn.Conv2d(width, width, 1)
        self.average = StatisticMambaPath(width, "average", d_state, d_conv, expand, backend)
        self.maximum = StatisticMambaPath(width, "maximum", d_state, d_conv, expand, backend)
        self.fuse = nn.Conv2d(2 * width, width, 1)

    def forward(self, x):
        p = self.project(x)
        output = self.fuse(torch.cat((self.average(p), self.maximum(p)), dim=1))
        return x + output if self.residual else output
