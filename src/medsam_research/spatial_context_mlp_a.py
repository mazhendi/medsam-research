"""Original SpatialContextA followed by residual channel refinement."""
from torch import nn

from .abc_decoder import SpatialContextA


class SpatialContextMLPA(SpatialContextA):
    """F=original_A(X); Y=F+MLP(F), with 2x channel expansion.

    Original spatial/context operators and their state keys are preserved.
    No outer X residual, LayerNorm, gamma, dropout or new attention.
    """
    def __init__(self, width, expansion=2):
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if not isinstance(expansion, int) or expansion < 1:
            raise ValueError("expansion must be a positive integer")
        super().__init__(width, residual=False)
        self.mlp = nn.Sequential(nn.Conv2d(width, width*expansion, 1), nn.GELU(),
                                 nn.Conv2d(width*expansion, width, 1))

    def forward(self, x):
        feature = super().forward(x)
        return feature + self.mlp(feature)
