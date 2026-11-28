"""Shallow queries attend to aligned multi-scale deep keys and values."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .multiscale_spatial_a import ChannelLayerNorm


class MultiscaleCrossAttentionA(nn.Module):
    """X + output(local_attention(Q(X), K(H), V(H))).

    H = LN(resize(fuse(cat(DW3/5/7(project(G)))))).
    Attention is over valid 5x5 neighbors only; no FFN or learned residual scale.
    Checkpoint the local operation during training to avoid retaining unfolded
    K/V neighborhoods in the backward graph. Each A has independent parameters.
    """
    def __init__(self, channels, context_channels, width=32, heads=4):
        super().__init__()
        if min(channels, context_channels, width, heads) < 1 or width % heads:
            raise ValueError('positive dimensions and width divisible by heads required')
        self.width, self.heads = width, heads
        self.norm_x = ChannelLayerNorm(channels)
        self.query = nn.Sequential(nn.Conv2d(channels, width, 1),
                                  nn.Conv2d(width, width, 3, padding=1, groups=width))
        self.context_project = nn.Conv2d(context_channels, width, 1)
        self.scales = nn.ModuleList(nn.Conv2d(width, width, k, padding=k//2, groups=width)
                                    for k in (3, 5, 7))
        self.context_fuse = nn.Conv2d(3*width, width, 1)
        self.norm_h = ChannelLayerNorm(width)
        self.key = nn.Conv2d(width, width, 1)
        self.value = nn.Conv2d(width, width, 1)
        self.output = nn.Conv2d(width, channels, 1)

    def local_attention(self, q, k, v):
        b, c, h, w = q.shape
        d = c // self.heads
        query = q.reshape(b, self.heads, d, h*w)
        keys = F.unfold(k, 5, padding=2).reshape(b, self.heads, d, 25, h*w)
        scores = (query.unsqueeze(3)*keys).sum(2) * d**-0.5
        valid = F.unfold(q.new_ones(1, 1, h, w), 5, padding=2).bool()
        # FP32 softmax for half/bfloat inputs, preserve double for reference tests.
        if scores.dtype in (torch.float16, torch.bfloat16):
            scores = scores.float()
        weights = scores.masked_fill(~valid.unsqueeze(1), float('-inf')).softmax(dim=2)
        values = F.unfold(v, 5, padding=2).reshape(b, self.heads, d, 25, h*w)
        return (weights.to(v.dtype).unsqueeze(2)*values).sum(3).reshape(b, c, h, w)

    def forward(self, x, context):
        q = self.query(self.norm_x(x))
        p = self.context_project(context)
        h = self.context_fuse(torch.cat([branch(p) for branch in self.scales], dim=1))
        h = self.norm_h(F.interpolate(h, size=x.shape[-2:], mode='bilinear', align_corners=False))
        k, v = self.key(h), self.value(h)
        if self.training and torch.is_grad_enabled():
            result = checkpoint(self.local_attention, q, k, v, use_reentrant=False)
        else:
            result = self.local_attention(q, k, v)
        return x + self.output(result)
