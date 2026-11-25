"""Candidate C: bidirectional local cross-attention before the original selector."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .abc_decoder import ChannelSpatialC
from .subband_decoder import resize


def _projection(width):
    return nn.Sequential(nn.Conv2d(width, width, 1),
                         nn.Conv2d(width, width, 3, padding=1, groups=width))


class LocalCrossAttention(nn.Module):
    """Single-head spatial attention over 9 neighbors; independent Q/K/V.

    Each query location receives values from the OTHER stream's neighborhood.
    Padding positions are masked, not treated as zero-valued valid tokens.
    """
    def __init__(self, width):
        super().__init__()
        self.query = _projection(width)
        self.key = _projection(width)
        self.value = _projection(width)
        self.scale = 1 / math.sqrt(width)

    def forward(self, query_source, context):
        batch, channels, height, width = query_source.shape
        q = self.query(query_source).flatten(2)
        k = F.unfold(self.key(context), 3, padding=1).reshape(batch, channels, 9, -1)
        v = F.unfold(self.value(context), 3, padding=1).reshape(batch, channels, 9, -1)
        # Softmax/logits in float32 under mixed precision; preserve float64 tests.
        dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        scores = (q.to(dtype).unsqueeze(2) * k.to(dtype)).sum(dim=1) * self.scale
        valid = F.unfold(query_source.new_ones(1, 1, height, width), 3, padding=1).bool()
        scores = scores.masked_fill(~valid, float('-inf'))
        weights = scores.softmax(dim=1).to(v.dtype)
        return (weights.unsqueeze(1) * v).sum(dim=2).reshape(batch, channels, height, width)


class BidirectionalLocalC(ChannelSpatialC):
    """Original C projections/selector, with symmetric local interaction added.

    Each stream concatenates a DW3 local feature with the cross-stream message,
    then PW(2C→2C), GELU, PW(2C→C). No outer residual or extra prediction head.
    """
    def __init__(self, width):
        super().__init__(width)
        self.u_to_e = LocalCrossAttention(width)
        self.e_to_u = LocalCrossAttention(width)
        self.local_e = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.local_u = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.mlp_e = nn.Sequential(nn.Conv2d(2 * width, 2 * width, 1), nn.GELU(),
                                   nn.Conv2d(2 * width, width, 1))
        self.mlp_u = nn.Sequential(nn.Conv2d(2 * width, 2 * width, 1), nn.GELU(),
                                   nn.Conv2d(2 * width, width, 1))

    def prepare(self, fine, deep):
        e, u = self.fine(fine), self.deep(resize(deep, fine))
        e_star = self.mlp_e(torch.cat((self.local_e(e), self.u_to_e(e, u)), 1))
        u_star = self.mlp_u(torch.cat((self.local_u(u), self.e_to_u(u, e)), 1))
        feature = self.interaction(torch.cat((e_star, u_star), 1))
        batch, channels, height, width = feature.shape
        global_logits = self.global_select(feature).reshape(batch, 2, channels, 1, 1)
        local_logits = self.local_select(feature).reshape(batch, 2, 1, height, width)
        weights = (global_logits + local_logits).softmax(1)
        return e_star, u_star, weights
