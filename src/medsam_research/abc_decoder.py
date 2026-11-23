"""Latest A/B/C design: fine A, coarse B, three independent C decoder stages."""
import torch
from torch import nn
from torch.nn import functional as F

from .selective_mamba import SelectiveMamba1
from .subband_decoder import resize


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_abc_options(options):
    allowed = {"width", "a", "b", "c", "d_state", "d_conv", "expand", "backend"}
    if not isinstance(options, dict) or set(options)-allowed:
        raise ValueError("Unknown ABC decoder options")
    width = options.get("width", 32)
    if not _positive_integer(width) or width < 8 or width % 4:
        raise ValueError("ABC width must be a multiple of four >=8")
    for key in ("d_state", "d_conv", "expand"):
        if not _positive_integer(options.get(key, {"d_state":16, "d_conv":4, "expand":2}[key])):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("a", "b", "c"):
        if not isinstance(options.get(key, True), bool):
            raise ValueError("ABC module switches must be boolean")
    if options.get("backend", "auto") not in {"auto", "torch", "fused"}:
        raise ValueError("Invalid Mamba backend")


class SpatialContextA(nn.Module):
    """STPNet-inspired topology; dense DW3/5/7 replace large dilation branches."""
    def __init__(self, width):
        super().__init__()
        self.project = nn.Conv2d(width, width, 1)
        self.spatial = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width),
                                     nn.Conv2d(width, width, 1), nn.Sigmoid())
        self.context = nn.ModuleList(nn.Conv2d(width, width, k, padding=k//2, groups=width)
                                     for k in (3, 5, 7))
        self.context_mix = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.fuse = nn.Conv2d(width*2, width, 1)

    def forward(self, x):
        projected = self.project(x)
        spatial = self.spatial(projected)*projected
        contextual = self.context_mix(sum(branch(projected) for branch in self.context))*projected
        return x+self.fuse(torch.cat((spatial, contextual), 1))


class LocalMambaB(nn.Module):
    """Four channel groups, DWConv→LN→Mamba-1→PW, progressive group interactions.

    Each mixer scans a raster-serialized feature map in ONE direction. This is
    explicitly not SS2D, bidirectional Mamba or an official HAMB reproduction.
    """
    def __init__(self, width, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        group = width//4
        self.project = nn.Conv2d(width, width, 1)
        self.local = nn.ModuleList(nn.Conv2d(group, group, k, padding=k//2, groups=group)
                                   for k in (3, 5, 7, 9))
        self.norm = nn.ModuleList(nn.LayerNorm(group) for _ in range(4))
        self.mamba = nn.ModuleList(SelectiveMamba1(group, d_state, d_conv, expand, backend)
                                   for _ in range(4))
        self.mix = nn.ModuleList(nn.Conv2d(group, group, 1) for _ in range(4))
        self.fuse = nn.Conv2d(width, width, 1)

    def forward(self, x):
        groups = self.project(x).chunk(4, 1)
        outputs = []
        for index, group in enumerate(groups):
            if outputs:
                group = group+outputs[-1]
            local = self.local[index](group)
            batch, channels, height, width = local.shape
            tokens = self.norm[index](local.flatten(2).transpose(1, 2))
            mixed = self.mamba[index](tokens).transpose(1, 2).reshape(batch, channels, height, width)
            outputs.append(self.mix[index](mixed))
        return x+self.fuse(torch.cat(outputs, 1))


class ChannelSpatialC(nn.Module):
    """Accepted C: global channel + local spatial logits, softmax over sources."""
    def __init__(self, width):
        super().__init__()
        hidden = max(8, width//4)
        self.fine = nn.Conv2d(width, width, 1)
        self.deep = nn.Conv2d(width, width, 1)
        self.interaction = nn.Sequential(nn.Conv2d(width*2, width, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(width), nn.ReLU())
        self.global_select = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(width, hidden, 1), nn.ReLU(),
                                           nn.Conv2d(hidden, width*2, 1))
        self.local_select = nn.Sequential(nn.Conv2d(width, hidden, 1), nn.ReLU(),
                                          nn.Conv2d(hidden, 2, 1))
        self.output = nn.Conv2d(width, width, 1)

    def prepare(self, fine, deep):
        x, y = self.fine(fine), self.deep(resize(deep, fine))
        feature = self.interaction(torch.cat((x, y), 1))
        batch, channels, height, width = feature.shape
        global_logits = self.global_select(feature).reshape(batch, 2, channels, 1, 1)
        local_logits = self.local_select(feature).reshape(batch, 2, 1, height, width)
        weights = (global_logits+local_logits).softmax(1)
        return x, y, weights

    def forward(self, fine, deep):
        x, y, weights = self.prepare(fine, deep)
        return self.output(weights[:, 0]*x+weights[:, 1]*y)


class LocalFusionControl(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(width*2, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(), nn.Conv2d(width, width, 1))

    def forward(self, fine, deep):
        return self.fuse(torch.cat((fine, resize(deep, fine)), 1))


class ABCPVT(nn.Module):
    def __init__(self, backbone, width=32, a=True, b=True, c=True,
                 d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        validate_abc_options(dict(width=width, a=a, b=b, c=c, d_state=d_state,
                                  d_conv=d_conv, expand=expand, backend=backend))
        self.backbone = backbone
        self.project = nn.ModuleList(nn.Conv2d(ch, width, 1) for ch in (64, 128, 320, 512))
        self.fine_blocks = nn.ModuleList(SpatialContextA(width) if a else nn.Identity()
                                         for _ in range(2))
        self.coarse_blocks = nn.ModuleList(
            LocalMambaB(width, d_state, d_conv, expand, backend) if b else nn.Identity()
            for _ in range(2))
        self.initialize = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.decoder = nn.ModuleList((ChannelSpatialC if c else LocalFusionControl)(width)
                                     for _ in range(3))
        self.final_head = nn.Conv2d(width, 1, 1)
        self.aux_heads = nn.ModuleList(nn.Conv2d(width, 1, 1) for _ in range(2))

    def forward(self, image):
        features = [project(feature) for project, feature in zip(self.project, self.backbone(image))]
        e1, e2 = [block(feature) for block, feature in zip(self.fine_blocks, features[:2])]
        e3, e4 = [block(feature) for block, feature in zip(self.coarse_blocks, features[2:])]
        d4 = self.initialize(e4)
        d3 = self.decoder[2](e3, d4)
        d2 = self.decoder[1](e2, d3)
        d1 = self.decoder[0](e1, d2)

        def up(logits):
            return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)

        return dict(final=up(self.final_head(d1)), aux2=up(self.aux_heads[0](d2)),
                    aux3=up(self.aux_heads[1](d3)))


def abc_loss(outputs, target, aux_weight=.5):
    from .sam2_unet_workflow import structure_loss

    return (structure_loss(outputs["final"], target)
            + aux_weight*(structure_loss(outputs["aux2"], target)
                          + structure_loss(outputs["aux3"], target)))
