"""Interface-controlled A/B/C experiments on the original Polyp-PVT topology.

All switches off reproduces the official graph and state-dict layout. All
variants retain original CFM, 44-grid heads, loss and sum-of-logits inference.
"""
import torch.nn.functional as F
from torch import nn

from .abc_decoder import ChannelSpatialC, LocalMambaB, SpatialContextA, validate_abc_options


def validate_slot_options(options):
    if not isinstance(options, dict):
        raise ValueError("slots must be a mapping")
    validate_abc_options(options)
    if options.get("width", 32) != 32:
        raise ValueError("Original Polyp-PVT interfaces require width=32")


class PolypPVTSlots(nn.Module):
    def __init__(self, baseline, a=False, b=False, c=False, width=32,
                 a_residual=False, b_residual=False, d_state=16, d_conv=4,
                 expand=2, backend="auto"):
        super().__init__()
        validate_slot_options(dict(a=a, b=b, c=c, width=width, a_residual=a_residual,
                                   b_residual=b_residual, d_state=d_state, d_conv=d_conv,
                                   expand=expand, backend=backend))
        self.use_a, self.use_c = a, c
        self.backbone = baseline.backbone
        for name in ("Translayer2_0", "Translayer2_1", "Translayer3_1", "Translayer4_1",
                     "CFM", "down05", "out_CFM", "out_SAM"):
            setattr(self, name, getattr(baseline, name))
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
        fused = self.CFM(p4, p3, p2)
        guidance = self.down05(self.Translayer2_0(fine))
        output = self.SAM(guidance, fused) if self.use_c else self.SAM(fused, guidance)
        first, second = self.out_CFM(fused), self.out_SAM(output)
        return (F.interpolate(first, scale_factor=8, mode="bilinear", align_corners=False),
                F.interpolate(second, scale_factor=8, mode="bilinear", align_corners=False))
