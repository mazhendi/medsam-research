"""Balanced refinement of A; Star-inspired interaction, not a StarNet replica."""
import torch
from torch import nn


class StarChannelInteraction(nn.Module):
    """Independent pointwise transforms, ReLU6 product, output projection."""

    def __init__(self, width):
        super().__init__()
        self.first = nn.Conv2d(width, width, 1)
        self.second = nn.Conv2d(width, width, 1)
        self.activation = nn.ReLU6()
        self.output = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return self.output(self.activation(self.first(x)) * self.second(x))


class _SelectionBranch(nn.Module):
    def __init__(self, width, kernels):
        super().__init__()
        self.filters = nn.ModuleList(
            nn.Conv2d(width, width, k, padding=k // 2, groups=width)
            for k in kernels
        )
        self.interaction = StarChannelInteraction(width)

    def forward(self, projected):
        feature = sum(layer(projected) for layer in self.filters) / len(self.filters)
        return self.interaction(feature).sigmoid() * projected


class BalancedStarA(nn.Module):
    """Local DW3/5 and context DW5/7, independent Star gates, concat fusion.

    No outer residual, no subtraction, no deformable sampling. All tensors keep
    the input grid and channel count. The two DW5 filters do not share weights.
    """

    def __init__(self, width):
        super().__init__()
        self.project = nn.Conv2d(width, width, 1)
        self.spatial = _SelectionBranch(width, (3, 5))
        self.context = _SelectionBranch(width, (5, 7))
        self.fuse = nn.Conv2d(2 * width, width, 1)

    def forward(self, x):
        projected = self.project(x)
        return self.fuse(torch.cat((self.spatial(projected), self.context(projected)), dim=1))
