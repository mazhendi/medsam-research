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
    if options.get("a1_variant", "original") not in {"original", "balanced_star"}:
        raise ValueError("a1_variant must be original or balanced_star")
    if options.get("a1_variant", "original") != "original":
        if not options.get("a", False):
            raise ValueError("balanced_star A1 requires a=True")
        if options.get("a_residual", False):
            raise ValueError("balanced_star A1 has no outer residual; set a_residual=False")
    if options.get("b_scan", "raster") not in {"raster", "axial"}:
        raise ValueError("b_scan must be raster or axial")
    if options.get("fusion", "aligned") not in {"aligned", "progressive"}:
        raise ValueError("fusion must be aligned or progressive")
    if options.get("b_scan", "raster") == "axial" and not options.get("b", False):
        raise ValueError("axial b_scan requires b=True")
    if options.get("fusion", "aligned") == "progressive" and options.get("cfm", True):
        raise ValueError("progressive fusion requires cfm=False")
    if not isinstance(options.get("cfm", True), bool):
        raise ValueError("cfm must be boolean")
    if not isinstance(options.get("deep_supervision", False), bool):
        raise ValueError("deep_supervision must be boolean")
    if not isinstance(options.get("a_f2", False), bool):
        raise ValueError("a_f2 must be boolean")
    if options.get("a_f2", False) and not options.get("a", False):
        raise ValueError("a_f2 requires a=True")
    validate_abc_options({key: value for key, value in options.items() if key not in {"cfm", "a_f2", "deep_supervision", "b_scan", "fusion", "a1_variant"}})
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
                 expand=2, backend="auto", cfm=True, a_f2=False, deep_supervision=False, b_scan="raster", fusion="aligned", a1_variant="original"):
        super().__init__()
        validate_slot_options(dict(a=a, b=b, c=c, width=width, a_residual=a_residual,
                                   b_residual=b_residual, d_state=d_state, d_conv=d_conv,
                                   expand=expand, backend=backend, cfm=cfm, a_f2=a_f2, deep_supervision=deep_supervision, b_scan=b_scan, fusion=fusion, a1_variant=a1_variant))
        self.use_a, self.use_c, self.use_cfm = a, c, cfm
        self.deep_supervision = deep_supervision
        self.backbone = baseline.backbone
        for name in ("Translayer2_0", "Translayer2_1", "Translayer3_1", "Translayer4_1",
                     "down05", "out_SAM"):
            setattr(self, name, getattr(baseline, name))
        if deep_supervision:
            self.aux3 = nn.Conv2d(32, 1, 1)
            self.aux4 = nn.Conv2d(32, 1, 1)
        else:
            self.out_CFM = baseline.out_CFM
        if cfm:
            self.CFM = baseline.CFM
        else:
            if fusion == "progressive":
                from .axial_progressive import ProgressiveFusion
                self.aligned_fusion = ProgressiveFusion(width)
            else:
                self.aligned_fusion = AlignedConcatFusion(width)
        if a:
            if a1_variant == "balanced_star":
                from .balanced_star_a import BalancedStarA
                self.cim = BalancedStarA(64)
            else:
                self.cim = SpatialContextA(64, residual=a_residual)
        else:
            self.ca, self.sa = baseline.ca, baseline.sa
        if a_f2:
            self.a2 = SpatialContextA(128, residual=a_residual)
        if b:
            from .axial_progressive import AxialLocalMambaB
            block = AxialLocalMambaB if b_scan == "axial" else LocalMambaB
            self.b3 = block(32, d_state, d_conv, expand, backend, residual=b_residual)
            self.b4 = block(32, d_state, d_conv, expand, backend, residual=b_residual)
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
        p2 = self.Translayer2_1(self.a2(f2) if hasattr(self, "a2") else f2)
        p3 = self.b3(self.Translayer3_1(f3))
        p4 = self.b4(self.Translayer4_1(f4))
        fused = self.CFM(p4, p3, p2) if self.use_cfm else self.aligned_fusion(p4, p3, p2)
        guidance = self.down05(self.Translayer2_0(fine))
        output = self.SAM(guidance, fused) if self.use_c else self.SAM(fused, guidance)
        if self.deep_supervision:
            result = {"final": F.interpolate(self.out_SAM(output), size=image.shape[-2:],
                                               mode="bilinear", align_corners=False)}
            if self.training:
                result["aux3"] = F.interpolate(self.aux3(p3), size=image.shape[-2:],
                                                mode="bilinear", align_corners=False)
                result["aux4"] = F.interpolate(self.aux4(p4), size=image.shape[-2:],
                                                mode="bilinear", align_corners=False)
            return result
        first, second = self.out_CFM(fused), self.out_SAM(output)
        return (F.interpolate(first, scale_factor=8, mode="bilinear", align_corners=False),
                F.interpolate(second, scale_factor=8, mode="bilinear", align_corners=False))


def slots_loss(predictions, target, aux_weights=(.5, .5)):
    """Legacy dual-head loss or final + two independent deep supervision losses."""
    from .sam2_unet_workflow import structure_loss
    if isinstance(predictions, dict):
        return (structure_loss(predictions["final"], target)
                + aux_weights[0] * structure_loss(predictions["aux3"], target)
                + aux_weights[1] * structure_loss(predictions["aux4"], target))
    return sum(structure_loss(logits, target) for logits in predictions)
