"""Multi-scale spatial mixing A, matching the approved two-residual diagram."""
import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each spatial position."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class MultiscaleSpatialA(nn.Module):
    """LN/PW → parallel DW3/5/7 → concat/PW → scaled skip; LN/MLP → scaled skip.

    Two intrinsic residuals are always present. The legacy a_residual flag
    must remain False to avoid implying an additional wrapper residual.
    Branches contain no extra activation, normalization, attention or pooling.
    """
    def __init__(self, channels, expansion=2, layer_scale_init=0.01):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not isinstance(expansion, int) or expansion < 1:
            raise ValueError("expansion must be a positive integer")
        self.norm1 = ChannelLayerNorm(channels)
        self.project = nn.Conv2d(channels, channels, 1)
        self.local = nn.ModuleList(nn.Conv2d(channels, channels, k, padding=k//2, groups=channels)
                                   for k in (3, 5, 7))
        self.fuse = nn.Conv2d(channels*3, channels, 1)
        self.gamma1 = nn.Parameter(torch.full((1, channels, 1, 1), layer_scale_init))
        self.norm2 = ChannelLayerNorm(channels)
        self.mlp = nn.Sequential(nn.Conv2d(channels, channels*expansion, 1), nn.GELU(),
                                 nn.Conv2d(channels*expansion, channels, 1))
        self.gamma2 = nn.Parameter(torch.full((1, channels, 1, 1), layer_scale_init))

    def forward(self, x):
        p = self.project(self.norm1(x))
        spatial = self.fuse(torch.cat([branch(p) for branch in self.local], dim=1))
        z = x + self.gamma1*spatial
        return z + self.gamma2*self.mlp(self.norm2(z))


class MultiscaleSpatialANoScale(MultiscaleSpatialA):
    """Same operators as MultiscaleSpatialA, with direct residual additions.

    No gamma parameters or fixed residual scaling. Historical scaled class
    and its checkpoint keys remain unchanged.
    """
    def __init__(self, channels, expansion=2):
        super().__init__(channels, expansion)
        del self.gamma1
        del self.gamma2

    def forward(self, x):
        p = self.project(self.norm1(x))
        spatial = self.fuse(torch.cat([branch(p) for branch in self.local], dim=1))
        z = x + spatial
        return z + self.mlp(self.norm2(z))
