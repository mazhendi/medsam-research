"""Candidate B: independent row/column sequences with tied bidirectional Mamba."""
import torch
from torch import nn

from .selective_mamba import SelectiveMamba1


class TiedBidirectionalMamba(nn.Module):
    """One parameter set, independent forward/reverse states, concat projection."""
    def __init__(self, width, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        self.mamba = SelectiveMamba1(width, d_state, d_conv, expand, backend)
        self.projection = nn.Linear(2 * width, width)

    def forward(self, tokens):
        batch = tokens.shape[0]
        # Batch stacking avoids cross-direction state transfer; sequence length
        # is unchanged. Reverse OUTPUT back before concatenating channels.
        forward, backward = self.mamba(
            torch.cat((tokens, tokens.flip(1)), dim=0)
        ).split(batch, dim=0)
        return self.projection(torch.cat((forward, backward.flip(1)), dim=-1))


class DualAxisMambaB(nn.Module):
    """Channel LN → PW → independent horizontal/vertical bi-Mamba → PW + X.

    Each row/column is an independent zero-state sequence. Within an axis,
    directions share weights. Between axes, and between B3/B4, weights differ.
    No 2D spatial convolution, multi-kernel branch, or inter-group addition.
    """
    def __init__(self, width=32, d_state=16, d_conv=4, expand=2, backend="auto", residual=True):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(width)
        self.project = nn.Conv2d(width, width, 1)
        self.horizontal = TiedBidirectionalMamba(width, d_state, d_conv, expand, backend)
        self.vertical = TiedBidirectionalMamba(width, d_state, d_conv, expand, backend)
        self.fuse = nn.Conv2d(2 * width, width, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        normalized = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        p = self.project(normalized)
        rows = p.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        columns = p.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        h = self.horizontal(rows).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        v = self.vertical(columns).reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        output = self.fuse(torch.cat((h, v), dim=1))
        return x + output if self.residual else output
