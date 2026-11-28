"""Original spatial gate with shared-QKV directional neighborhood attention."""
import torch
import torch.nn.functional as F
from torch import nn


class DirectionalNeighborhood(nn.Module):
    """Center queries attend to valid neighbors only; no wraparound or pooling."""
    def __init__(self, channels, kernel, heads=4):
        super().__init__()
        if channels < 1 or heads < 1 or channels % heads:
            raise ValueError("channels must be positive and divisible by heads")
        if any(k < 1 or k % 2 == 0 for k in kernel):
            raise ValueError("kernel dimensions must be positive odd integers")
        self.kernel, self.heads = tuple(kernel), heads
        self.scale = (channels // heads)**-0.5
        self.bias = nn.Parameter(torch.zeros(heads, kernel[0]*kernel[1]))

    def forward(self, q, k, v):
        batch, channels, height, width = q.shape
        count = self.kernel[0]*self.kernel[1]
        length = height*width
        padding = (self.kernel[0]//2, self.kernel[1]//2)
        query = q.reshape(batch, self.heads, channels//self.heads, length)
        keys = F.unfold(k, self.kernel, padding=padding).reshape(
            batch, self.heads, channels//self.heads, count, length)
        scores = torch.einsum("bhdl,bhdnl->bhnl", query, keys)*self.scale
        scores = scores + self.bias[None,:,:,None]
        valid = F.unfold(q.new_ones(1,1,height,width), self.kernel, padding=padding).bool()
        scores = scores.masked_fill(~valid[:,None], float("-inf"))
        weights = scores.softmax(dim=2)
        values = F.unfold(v, self.kernel, padding=padding).reshape(
            batch, self.heads, channels//self.heads, count, length)
        out = torch.einsum("bhnl,bhdnl->bhdl", weights, values)
        return out.reshape(batch, channels, height, width)


class DirectionalContextA(nn.Module):
    """No MLP, normalization, gamma or outer residual. No spatial downsampling.

    Q/K/V are shared by all three directions; each has its own relative bias.
    Spatial gate and final modulation/fusion follow the original A.
    """
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels < 1 or channels % heads:
            raise ValueError("channels must be positive and divisible by heads")
        self.project = nn.Conv2d(channels, channels, 1)
        self.spatial = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                     nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.qkv = nn.Conv2d(channels, 3*channels, 1)
        self.neighborhoods = nn.ModuleList(DirectionalNeighborhood(channels, kernel, heads)
                                          for kernel in ((1,7),(7,1),(3,3)))
        self.context_fuse = nn.Conv2d(3*channels, channels, 1)
        self.context_mix = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.fuse = nn.Conv2d(2*channels, channels, 1)

    def forward(self, x):
        p = self.project(x)
        spatial = self.spatial(p)*p
        q, k, v = self.qkv(p).chunk(3, dim=1)
        contexts = [layer(q,k,v) for layer in self.neighborhoods]
        contextual = self.context_mix(self.context_fuse(torch.cat(contexts,dim=1)))*p
        return self.fuse(torch.cat((spatial,contextual),dim=1))
