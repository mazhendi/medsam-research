"""Opt-in axial row/column scans and adjacent-level fusion for PVT slots.

Not standard SS2D: each row/column is an independent zero-state sequence.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .abc_decoder import LocalMambaB


def axial_bidirectional_mix(feature, norm, mixer):
    """Share one mixer over four directions, restore spatial order, then mean."""
    batch, channels, height, width = feature.shape
    grid = norm(feature.permute(0, 2, 3, 1))
    rows = grid.reshape(batch * height, width, channels)
    columns = grid.permute(0, 2, 1, 3).reshape(batch * width, height, channels)
    row_forward = mixer(rows).reshape(batch, height, width, channels)
    row_backward = mixer(rows.flip(1)).flip(1).reshape(batch, height, width, channels)
    col_forward = mixer(columns).reshape(batch, width, height, channels).permute(0, 2, 1, 3)
    col_backward = mixer(columns.flip(1)).flip(1).reshape(batch, width, height, channels).permute(0, 2, 1, 3)
    return ((row_forward + row_backward + col_forward + col_backward) * .25).permute(0, 3, 1, 2).contiguous()


class AxialLocalMambaB(LocalMambaB):
    """Retain kernels/groups/progressive links; change only spatial scan layout."""
    def forward(self, x):
        groups = self.project(x).chunk(4, 1)
        outputs = []
        for index, group in enumerate(groups):
            if outputs:
                group = group + outputs[-1]
            local = self.local[index](group)
            mixed = axial_bidirectional_mix(local, self.norm[index], self.mamba[index])
            outputs.append(self.mix[index](mixed))
        output = self.fuse(torch.cat(outputs, 1))
        return x + output if self.residual else output


class AdjacentFusion(nn.Module):
    """Resize coarse -> concat with finer -> PW -> DW3 -> GN -> GELU."""
    def __init__(self, width=32):
        super().__init__()
        self.mix = nn.Sequential(nn.Conv2d(width * 2, width, 1),
                                 nn.Conv2d(width, width, 3, padding=1, groups=width),
                                 nn.GroupNorm(8, width), nn.GELU())

    def forward(self, coarse, finer):
        coarse = F.interpolate(coarse, size=finer.shape[-2:], mode="bilinear", align_corners=True)
        return self.mix(torch.cat((coarse, finer), 1))


class ProgressiveFusion(nn.Module):
    """J3(P4,P3) -> T3; J2(T3,P2) -> U, independent unit parameters."""
    def __init__(self, width=32):
        super().__init__()
        self.j3 = AdjacentFusion(width)
        self.j2 = AdjacentFusion(width)

    def forward(self, deep, middle, fine):
        return self.j2(self.j3(deep, middle), fine)
