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
    if options.get("b_variant", "original") not in {"original", "cross_scale_b1", "shared_mamba", "dual_axis", "multiscale_readwrite", "dual_statistic", "progressive_concat"}:
        raise ValueError("b_variant must be original, cross_scale_b1, shared_mamba, dual_axis, multiscale_readwrite, dual_statistic or progressive_concat")
    if options.get("b_variant", "original") == "cross_scale_b1":
        if not options.get("b", False) or options.get("ablation", "none") == "wo_b":
            raise ValueError("B1 requires B enabled (not wo_b)")
        if options.get("b_residual", False) or options.get("b_scan", "raster") != "raster":
            raise ValueError("B1 requires b_residual=False and b_scan=raster")
    if options.get("b_variant", "original") == "shared_mamba":
        if not options.get("b", False) or options.get("ablation", "none") == "wo_b":
            raise ValueError("shared_mamba requires B enabled (not wo_b)")
        if options.get("b_scan", "raster") != "raster":
            raise ValueError("shared_mamba uses raster scans, not axial")
    if options.get("b_variant", "original") == "dual_axis":
        if not options.get("b", False) or options.get("ablation", "none") == "wo_b":
            raise ValueError("dual_axis requires B enabled (not wo_b)")
        if options.get("b_scan", "raster") != "raster":
            raise ValueError("dual_axis uses its own scan; do not set legacy b_scan=axial")
    if options.get("b_variant", "original") == "multiscale_readwrite":
        if not options.get("b", False) or options.get("ablation", "none") == "wo_b":
            raise ValueError("multiscale_readwrite requires B enabled (not wo_b)")
        if options.get("b_scan", "raster") != "raster":
            raise ValueError("multiscale_readwrite uses raster scans")
    if options.get("b_variant", "original") == "dual_statistic":
        if not options.get("b", False) or options.get("ablation", "none") == "wo_b":
            raise ValueError("dual_statistic requires B enabled (not wo_b)")
        if options.get("b_scan", "raster") != "raster":
            raise ValueError("dual_statistic uses raster scans")
    if options.get("b_variant", "original") == "progressive_concat":
        if not options.get("b", False) or options.get("ablation", "none") == "wo_b":
            raise ValueError("progressive_concat requires B enabled (not wo_b)")
        if options.get("b_residual", False) or options.get("b_scan", "raster") != "raster":
            raise ValueError("progressive_concat requires b_residual=False and b_scan=raster")
    if options.get("c_variant", "original") not in {"original", "bidirectional_local"}:
        raise ValueError("c_variant must be original or bidirectional_local")
    if options.get("c_variant", "original") != "original":
        if not options.get("c", False) or options.get("ablation", "none") == "wo_c":
            raise ValueError("bidirectional_local requires C enabled (not wo_c)")
    ablation = options.get("ablation", "none")
    if ablation not in {"none", "wo_a", "wo_b", "wo_c"}:
        raise ValueError("ablation must be none, wo_a, wo_b or wo_c")
    if ablation != "none":
        if not all(options.get(k, False) for k in ("a", "a_f2", "b", "c", "deep_supervision")) or options.get("cfm", True):
            raise ValueError("w/o ablations require full A/B/C, a_f2, three heads and cfm=False")
    if options.get("a1_variant", "original") not in {"original", "balanced_star", "dual_stream_gated", "multiscale_spatial", "multiscale_spatial_no_scale", "original_mlp", "directional_context", "multiscale_cross_attention", "dual_source_cross_attention"}:
        raise ValueError("a1_variant must be original, balanced_star, dual_stream_gated, multiscale_spatial, multiscale_spatial_no_scale, original_mlp, directional_context, multiscale_cross_attention or dual_source_cross_attention")
    if options.get("a1_variant", "original") != "original":
        if not options.get("a", False):
            raise ValueError("non-original A1 requires a=True")
        if options.get("a_residual", False):
            raise ValueError("Set a_residual=False for non-original A1; multiscale_spatial already has two intrinsic residuals")
    if options.get("a2_variant", "original") not in {"original", "balanced_star", "dual_stream_gated", "multiscale_spatial", "multiscale_spatial_no_scale", "original_mlp", "directional_context", "multiscale_cross_attention", "dual_source_cross_attention"}:
        raise ValueError("a2_variant must be original, balanced_star, dual_stream_gated, multiscale_spatial, multiscale_spatial_no_scale, original_mlp, directional_context, multiscale_cross_attention or dual_source_cross_attention")
    if options.get("a2_variant", "original") != "original":
        if not options.get("a_f2", False):
            raise ValueError("non-original A2 requires a_f2=True")
        if options.get("a_residual", False):
            raise ValueError("Set a_residual=False for non-original A2; multiscale_spatial already has two intrinsic residuals")
    if options.get("b_scan", "raster") not in {"raster", "axial"}:
        raise ValueError("b_scan must be raster or axial")
    if options.get("fusion", "aligned") not in {"aligned", "progressive", "dual_path"}:
        raise ValueError("fusion must be aligned, progressive or dual_path")
    if options.get("b_scan", "raster") == "axial" and not options.get("b", False):
        raise ValueError("axial b_scan requires b=True")
    if options.get("fusion", "aligned") == "progressive" and options.get("cfm", True):
        raise ValueError("progressive fusion requires cfm=False")
    if options.get("fusion", "aligned") == "dual_path":
        if options.get("cfm", True) or not all(options.get(k, False) for k in ("a", "a_f2", "b", "c", "deep_supervision")):
            raise ValueError("dual_path requires A1/A2/B/C, three heads, cfm=False")
        if options.get("ablation", "none") != "none":
            raise ValueError("dual_path does not support legacy deletion ablations")
    if not isinstance(options.get("cfm", True), bool):
        raise ValueError("cfm must be boolean")
    if not isinstance(options.get("deep_supervision", False), bool):
        raise ValueError("deep_supervision must be boolean")
    if not isinstance(options.get("a_f2", False), bool):
        raise ValueError("a_f2 must be boolean")
    if options.get("a_f2", False) and not options.get("a", False):
        raise ValueError("a_f2 requires a=True")
    validate_abc_options({key: value for key, value in options.items() if key not in {"cfm", "a_f2", "deep_supervision", "b_scan", "fusion", "a1_variant", "a2_variant", "ablation", "c_variant", "b_variant"}})
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


class PlainConcatC(nn.Module):
    """w/o C interface adapter: no attention, normalization or spatial mixing."""
    def __init__(self, width=32):
        super().__init__()
        self.projection = nn.Conv2d(2 * width, width, 1)

    def forward(self, fine, deep):
        deep = F.interpolate(deep, size=fine.shape[-2:], mode="bilinear", align_corners=True)
        return self.projection(torch.cat((fine, deep), dim=1))


class PolypPVTSlots(nn.Module):
    def __init__(self, baseline, a=False, b=False, c=False, width=32,
                 a_residual=False, b_residual=False, d_state=16, d_conv=4,
                 expand=2, backend="auto", cfm=True, a_f2=False, deep_supervision=False, b_scan="raster", fusion="aligned", a1_variant="original", a2_variant="original", ablation="none", c_variant="original", b_variant="original"):
        super().__init__()
        validate_slot_options(dict(a=a, b=b, c=c, width=width, a_residual=a_residual,
                                   b_residual=b_residual, d_state=d_state, d_conv=d_conv,
                                   expand=expand, backend=backend, cfm=cfm, a_f2=a_f2, deep_supervision=deep_supervision, b_scan=b_scan, fusion=fusion, a1_variant=a1_variant, a2_variant=a2_variant, ablation=ablation, c_variant=c_variant, b_variant=b_variant))
        self.a1_cross = a1_variant in {"multiscale_cross_attention", "dual_source_cross_attention"}
        self.a2_cross = a2_variant in {"multiscale_cross_attention", "dual_source_cross_attention"}
        self.fusion = fusion
        self.ablation = ablation
        if ablation == "wo_a":
            a, a_f2 = False, False
        elif ablation == "wo_b":
            b = False
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
            if fusion == "dual_path":
                self.detail_fusion = nn.Conv2d(2*width, width, 1)
                self.context_fusion = nn.Conv2d(2*width, width, 1)
            elif fusion == "progressive":
                from .axial_progressive import ProgressiveFusion
                self.aligned_fusion = ProgressiveFusion(width)
            else:
                self.aligned_fusion = AlignedConcatFusion(width)
        if a:
            if a1_variant == "dual_source_cross_attention":
                from .dual_source_cross_attention_a import DualSourceCrossAttentionA
                self.cim = DualSourceCrossAttentionA(64, 128, width=32)
            elif a1_variant == "multiscale_cross_attention":
                from .multiscale_cross_attention_a import MultiscaleCrossAttentionA
                self.cim = MultiscaleCrossAttentionA(64, 128, width=32)
            elif a1_variant == "directional_context":
                from .directional_context_a import DirectionalContextA
                self.cim = DirectionalContextA(64)
            elif a1_variant == "original_mlp":
                from .spatial_context_mlp_a import SpatialContextMLPA
                self.cim = SpatialContextMLPA(64)
            elif a1_variant == "multiscale_spatial_no_scale":
                from .multiscale_spatial_a import MultiscaleSpatialANoScale
                self.cim = MultiscaleSpatialANoScale(64)
            elif a1_variant == "multiscale_spatial":
                from .multiscale_spatial_a import MultiscaleSpatialA
                self.cim = MultiscaleSpatialA(64)
            elif a1_variant == "dual_stream_gated":
                from .dual_stream_a import DualStreamGatedA
                self.cim = DualStreamGatedA(64)
            elif a1_variant == "balanced_star":
                from .balanced_star_a import BalancedStarA
                self.cim = BalancedStarA(64)
            else:
                self.cim = SpatialContextA(64, residual=a_residual)
        elif ablation != "wo_a":
            self.ca, self.sa = baseline.ca, baseline.sa
        if a_f2:
            if a2_variant == "dual_source_cross_attention":
                from .dual_source_cross_attention_a import DualSourceCrossAttentionA
                self.a2 = DualSourceCrossAttentionA(128, 320, width=64)
            elif a2_variant == "multiscale_cross_attention":
                from .multiscale_cross_attention_a import MultiscaleCrossAttentionA
                self.a2 = MultiscaleCrossAttentionA(128, 320, width=64)
            elif a2_variant == "directional_context":
                from .directional_context_a import DirectionalContextA
                self.a2 = DirectionalContextA(128)
            elif a2_variant == "original_mlp":
                from .spatial_context_mlp_a import SpatialContextMLPA
                self.a2 = SpatialContextMLPA(128)
            elif a2_variant == "multiscale_spatial_no_scale":
                from .multiscale_spatial_a import MultiscaleSpatialANoScale
                self.a2 = MultiscaleSpatialANoScale(128)
            elif a2_variant == "multiscale_spatial":
                from .multiscale_spatial_a import MultiscaleSpatialA
                self.a2 = MultiscaleSpatialA(128)
            elif a2_variant == "dual_stream_gated":
                from .dual_stream_a import DualStreamGatedA
                self.a2 = DualStreamGatedA(128)
            elif a2_variant == "balanced_star":
                from .balanced_star_a import BalancedStarA
                self.a2 = BalancedStarA(128)
            else:
                self.a2 = SpatialContextA(128, residual=a_residual)
        if b and b_variant == "progressive_concat":
            from .progressive_concat_b import ProgressiveConcatMambaB
            self.b3 = ProgressiveConcatMambaB(32, d_state, d_conv, expand, backend)
            self.b4 = ProgressiveConcatMambaB(32, d_state, d_conv, expand, backend)
        elif b and b_variant == "dual_statistic":
            from .dual_statistic_b import DualStatisticMambaB
            self.b3 = DualStatisticMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
            self.b4 = DualStatisticMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
        elif b and b_variant == "multiscale_readwrite":
            from .multiscale_readwrite_b import MultiscaleReadWriteB
            self.b3 = MultiscaleReadWriteB(32, d_state, d_conv, expand, backend, residual=b_residual)
            self.b4 = MultiscaleReadWriteB(32, d_state, d_conv, expand, backend, residual=b_residual)
        elif b and b_variant == "dual_axis":
            from .dual_axis_b import DualAxisMambaB
            self.b3 = DualAxisMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
            self.b4 = DualAxisMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
        elif b and b_variant == "shared_mamba":
            from .shared_mamba_b import SharedMambaB
            self.b3 = SharedMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
            self.b4 = SharedMambaB(32, d_state, d_conv, expand, backend, residual=b_residual)
        elif b and b_variant == "cross_scale_b1":
            from .cross_scale_b1 import CrossScaleMambaB1
            self.b3 = CrossScaleMambaB1(32, d_state, d_conv, expand, backend)
            self.b4 = CrossScaleMambaB1(32, d_state, d_conv, expand, backend)
        elif b:
            from .axial_progressive import AxialLocalMambaB
            block = AxialLocalMambaB if b_scan == "axial" else LocalMambaB
            self.b3 = block(32, d_state, d_conv, expand, backend, residual=b_residual)
            self.b4 = block(32, d_state, d_conv, expand, backend, residual=b_residual)
        else:
            self.b3, self.b4 = nn.Identity(), nn.Identity()
        if ablation == "wo_c":
            self.SAM = PlainConcatC(32)
        elif c and c_variant == "bidirectional_local":
            from .local_interaction_c import BidirectionalLocalC
            self.SAM = BidirectionalLocalC(32)
        else:
            self.SAM = ChannelSpatialC(32) if c else baseline.SAM

    def forward(self, image):
        f1, f2, f3, f4 = self.backbone(image)
        if self.use_a:
            fine = self.cim(f1, f2) if self.a1_cross else self.cim(f1)
        elif self.ablation == "wo_a":
            fine = f1
        else:
            fine = self.ca(f1)*f1
            fine = self.sa(fine)*fine
        enhanced_f2 = (self.a2(f2, f3) if self.a2_cross else self.a2(f2)) if hasattr(self, "a2") else f2
        p2 = self.Translayer2_1(enhanced_f2)
        p3 = self.b3(self.Translayer3_1(f3))
        p4 = self.b4(self.Translayer4_1(f4))
        if self.fusion == "dual_path":
            # Preserve F1 grid; F2 contributes only to the detail path.
            detail = self.Translayer2_0(fine)
            p2_up = F.interpolate(p2, size=detail.shape[-2:], mode="bilinear", align_corners=True)
            guidance = self.detail_fusion(torch.cat((detail, p2_up), dim=1))
            # Preserve native B3/B4 outputs for independent auxiliary heads.
            p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=True)
            context = self.context_fusion(torch.cat((p3, p4_up), dim=1))
            fused = F.interpolate(context, size=guidance.shape[-2:], mode="bilinear", align_corners=True)
        else:
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
