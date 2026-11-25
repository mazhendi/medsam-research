"""Multiscale shared-Mamba candidate; parameter sharing, independent states."""
import torch
from torch import nn

from .selective_mamba import SelectiveMamba1


class SharedMambaB(nn.Module):
    """Full-width DW3/5/7, batch-stack into one Mamba, concat + outer residual.

    Every path has `width` channels, not width/3. Batch stacking never joins
    sequence lengths. B3 and B4 instantiate separate blocks; sharing occurs
    only across the three scales WITHIN one block.
    """
    def __init__(self, width=32, d_state=16, d_conv=4, expand=2, backend="auto", residual=True):
        super().__init__()
        self.residual = residual
        self.project = nn.Conv2d(width, width, 1)
        self.local = nn.ModuleList(nn.Conv2d(width, width, k, padding=k // 2, groups=width)
                                   for k in (3, 5, 7))
        self.norm = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.mamba = SelectiveMamba1(width, d_state, d_conv, expand, backend)
        self.fuse = nn.Conv2d(3 * width, width, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        p = self.project(x)
        tokens = [norm(local(p).flatten(2).transpose(1, 2))
                  for local, norm in zip(self.local, self.norm)]
        # Three independent B-sized batches, ONE common Mamba parameter set.
        mixed = self.mamba(torch.cat(tokens, dim=0))
        restored = [part.transpose(1, 2).reshape(batch, channels, height, width)
                    for part in mixed.split(batch, dim=0)]
        output = self.fuse(torch.cat(restored, dim=1))
        return x + output if self.residual else output
