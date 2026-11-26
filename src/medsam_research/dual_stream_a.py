"""Single-input dual-stream gated interaction A; no outer residual."""
import torch
from torch import nn


class DualStreamGatedA(nn.Module):
    """Split projected channels; independently gate cross-stream messages.

    O1=MLP1(V1+W1*V2), O2=MLP2(V2+W2*V1).
    MLP expansion is 2x per half-width; gates are independent sigmoids.
    """
    def __init__(self, channels, expansion=2):
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("channels must be positive and even")
        if not isinstance(expansion, int) or expansion < 1:
            raise ValueError("expansion must be a positive integer")
        half = channels // 2
        self.project = nn.Conv2d(channels, channels, 1)
        self.gates = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(), nn.Conv2d(channels, channels, 1))
        def mlp():
            return nn.Sequential(nn.Conv2d(half, half*expansion, 1), nn.GELU(),
                                 nn.Conv2d(half*expansion, half, 1))
        self.mlp1, self.mlp2 = mlp(), mlp()
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        v1, v2 = self.project(x).chunk(2, dim=1)
        w1, w2 = self.gates(torch.cat((v1, v2), dim=1)).sigmoid().chunk(2, dim=1)
        o1 = self.mlp1(v1 + w1*v2)
        o2 = self.mlp2(v2 + w2*v1)
        return self.fuse(torch.cat((o1, o2), dim=1))
