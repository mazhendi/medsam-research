"""Interface-controlled A/B/C experiments on the original Polyp-PVT topology.

All A/B/C switches off with cfm=True reproduces the official graph.
cfm=False removes original CFM and uses only aligned concatenation + 1x1 fusion.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .abc_decoder import ChannelSpatialC, LocalMambaB, SpatialContextA, validate_abc_options


def validate_slot_options(options):
    if not isinstance(options, dict):
        raise ValueError("slots must be a mapping")
    if not isinstance(options.get("cfm", True), bool):
        raise ValueError("cfm must be boolean")
    validate_abc_options({key: value for key, value in options.items() if key != "cfm"})
    if options.get("width", 32) != 32:
        raise ValueError("Original Polyp-PVT interfaces require width=32")


class AlignedConcatFusion(nn.Module):
    """Minimal three-input alignment; no cascaded products or CFM layers."""
    def __init__(self, width=32):
        super().__init__()
        self.projection = nn.Conv2d(width*3, width, 1)

    def forward(self, deep, middle, fine):
        size = fine.shape[-2:]
        deep = F.interpolate(deep, size=size, mode="bilinear", align_corners=True)
        middle = F.interpolate(middle, size=size, mode="bilinear", align_corners=True)
        return self.projection(torch.cat((deep, middle, fine), 1))


class PolypPVTSlots(nn.Module):
    def __init__(self, baseline, a=False, b=False, c=False, width=32,
                 a_residual=False, b_residual=False, d_state=16, d_conv=4,
                 expand=2, backend="auto", cfm=True):
        super().__init__()
        validate_slot_options(dict(a=a, b=b, c=c, width=width, a_residual=a_residual,
                                   b_residual=b_residual, d_state=d_state, d_conv=d_conv,
                                   expand=expand, backend=backend, cfm=cfm))
        self.use_a, self.use_c, self.use_cfm = a, c, cfm
        self.backbone = baseline.backbone
        for name in ("Translayer2_0", "Translayer2_1", "Translayer3_1", "Translayer4_1",
                     "down05", "out_CFM", "out_SAM"):
            setattr(self, name, getattr(baseline, name))
        if cfm:
            self.CFM = baseline.CFM
        else:
            self.aligned_fusion = AlignedConcatFusion(width)
        if a:
            self.cim = SpatialContextA(64, residual=a_residual)
        else:
            self.ca, self.sa = baseline.ca, baseline.sa
        if b:
            self.b3 = LocalMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
            self.b4 = LocalMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
        else:
            self.b3, self.b4 = nn.Identity(), nn.Identity()
        self.SAM = ChannelSpatialC(32) if c else baseline.SAM

    def forward(self, image):
        f1, f2, f3, f4 = self.backbone(image)
        if self.use_a:
            fine = self.cim(f1)
        else:
            fine = self.ca(f1)*f1
            fine = self.sa(fine)*fine
        p2 = self.Translayer2_1(f2)
        p3 = self.b3(self.Translayer3_1(f3))
        p4 = self.b4(self.Translayer4_1(f4))
        fused = self.CFM(p4, p3, p2) if self.use_cfm else self.aligned_fusion(p4, p3, p2)
        guidance = self.down05(self.Translayer2_0(fine))
        output = self.SAM(guidance, fused) if self.use_c else self.SAM(fused, guidance)
        first, second = self.out_CFM(fused), self.out_SAM(output)
        return (F.interpolate(first, scale_factor=8, mode="bilinear", align_corners=False),
                F.interpolate(second, scale_factor=8, mode="bilinear", align_corners=False))
