"""Directional boundary contrast and boundary-constrained cross-scale fusion.

Research prototype: no accuracy or novelty claim. All geometry is predicted;
labels are used only by boundary_contrast_loss. SAM2 stays frozen.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .metrics import segmentation_loss

MODES = ("full", "no_contrast", "no_constraint", "equal_fusion", "no_boundary")


@dataclass
class BoundaryContrastConfig:
    width: int = 32
    radii: tuple = (1.0, 2.0, 4.0)
    normal_scale: float = 0.05
    affinity_strength: float = 4.0
    coarse_weight: float = 0.2
    mode: str = "full"
    architecture: str = "baseline_adapter"
    train_decoder: bool = False

    def validate(self):
        import math

        if type(self.width) is not int or self.width < 4:
            raise ValueError("width must be an integer >= 4")
        if not self.radii or any(not math.isfinite(r) or r <= 0 for r in self.radii):
            raise ValueError("radii must be finite and positive")
        for key in ("normal_scale", "affinity_strength", "coarse_weight"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key == "normal_scale" and value == 0):
                raise ValueError(f"Invalid {key}")
        if self.mode not in MODES:
            raise ValueError("Unknown boundary-contrast mode")
        if self.architecture not in ("baseline_adapter", "custom_decoder"):
            raise ValueError("Unknown decoder architecture")
        if type(self.train_decoder) is not bool:
            raise ValueError("train_decoder must be boolean")
        if self.architecture == "custom_decoder" and self.train_decoder:
            raise ValueError("Legacy custom decoder cannot fine-tune the SAM decoder")
        return self

    def to_dict(self):
        result = asdict(self)
        result["radii"] = list(self.radii)
        return result


def load_boundary_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(BoundaryContrastConfig)}:
        raise ValueError("Unknown boundary configuration fields")
    return BoundaryContrastConfig(**raw).validate()


def spatial_gradient(probability):
    padded = F.pad(probability.float(), (1, 1, 1, 1), mode="replicate")
    dx = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
    dy = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
    return dx, dy


def directional_pairs(feature, probability, radii, normal_scale):
    """Paired samples in FEATURE-grid pixels; outward n = -grad(p)/|grad(p)|.

    align_corners=False pixel centers, FP32 sampling for AMP stability. Invalid
    pairs outside the image are suppressed rather than treating padding as data.
    A flat p yields equal pairs and exactly zero directional reliability.
    """
    feature, probability = feature.float(), probability.float()
    b, _, height, width = feature.shape
    if probability.shape != (b, 1, height, width):
        raise ValueError("Probability and feature grids must agree")
    dx, dy = spatial_gradient(probability)
    magnitude = (dx.square() + dy.square() + 1e-12).sqrt()
    nx, ny = -dx / magnitude, -dy / magnitude
    strength = (dx.square() + dy.square()) / (dx.square() + dy.square() + normal_scale**2)
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=feature.device) + 0.5) * 2 / height - 1,
        (torch.arange(width, device=feature.device) + 0.5) * 2 / width - 1,
        indexing="ij",
    )
    grid = torch.stack((xx, yy), -1)[None].expand(b, -1, -1, -1)
    direction = torch.cat((nx * 2 / width, ny * 2 / height), 1).permute(0, 2, 3, 1)
    pairs, validity = [], []
    for radius in radii:
        inner, outer = grid - radius * direction, grid + radius * direction
        # A conservative radius margin is independent of predicted directions.
        # It avoids hard, data-dependent validity flips from floating-point noise
        # when a nearly axis-aligned pair lands exactly on a border pixel.
        pixel_x = torch.arange(width, device=feature.device)[None, :]
        pixel_y = torch.arange(height, device=feature.device)[:, None]
        valid = (
            (pixel_x >= radius)
            & (pixel_x <= width - 1 - radius)
            & (pixel_y >= radius)
            & (pixel_y <= height - 1 - radius)
        )[None, None]
        fi = F.grid_sample(
            feature, inner, mode="bilinear", padding_mode="border", align_corners=False
        )
        fo = F.grid_sample(
            feature, outer, mode="bilinear", padding_mode="border", align_corners=False
        )
        pairs.append((fi - fo) * valid)
        validity.append(valid.float())
    reliable = strength * torch.stack(validity).mean(0)
    return torch.cat(pairs, 1), reliable, torch.cat((nx, ny), 1)


class InnerOuterBoundary(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        w = config.width
        self.local = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1, groups=w),
            nn.Conv2d(w, w, 1),
            nn.GroupNorm(1, w),
            nn.GELU(),
        )
        self.contrast = nn.Sequential(
            nn.Conv2d(w * len(config.radii), w, 1), nn.GroupNorm(1, w), nn.GELU()
        )
        self.gate = nn.Conv2d(2 * w + 1, w, 1)

    def forward(self, feature, probability):
        local = self.local(feature)
        contrast, reliability, normal = directional_pairs(
            feature, probability, self.config.radii, self.config.normal_scale
        )
        contrast = self.contrast(contrast)
        gate = torch.sigmoid(self.gate(torch.cat((local, contrast, reliability), 1)))
        gate = reliability * gate
        boundary = (1 - gate) * local + gate * contrast
        if self.config.mode == "no_contrast":
            boundary = local
        if self.config.mode == "no_boundary":
            boundary = torch.zeros_like(local)
        return boundary, reliability, normal, gate


def neighbor(x, dy, dx):
    height, width = x.shape[-2:]
    return F.pad(x, (1, 1, 1, 1), mode="replicate")[
        :, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width
    ]


def constrained_pool(feature, probability, boundary_gate, strength):
    """3x3 normalized affinities penalize crossing predicted regional boundaries.

    Scalar learned boundary-gate differences supply additional local structure.
    No detached geometry: final losses can reach the coarse head through weights.
    """
    feature, probability, boundary_gate = (
        feature.float(),
        probability.float(),
        boundary_gate.float(),
    )
    numerator, denominator = torch.zeros_like(feature), torch.zeros_like(probability)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            weight = torch.exp(
                -strength * (probability - neighbor(probability, dy, dx)).abs()
                - (boundary_gate - neighbor(boundary_gate, dy, dx)).abs()
            )
            numerator = numerator + weight * neighbor(feature, dy, dx)
            denominator = denominator + weight
    return numerator / denominator.clamp_min(1e-6)


class BoundaryCrossScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        w = config.width
        self.detail_gate = nn.Conv2d(w, w, 1)
        self.boundary_gate = nn.Conv2d(w, 1, 1)
        self.semantic = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(1, w), nn.GELU())
        self.competition = nn.Conv2d(3 * w, w, 1)

    def forward(self, shallow, deep, boundary, probability):
        deep = F.interpolate(deep, shallow.shape[-2:], mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary, shallow.shape[-2:], mode="bilinear", align_corners=False)
        probability = F.interpolate(
            probability, shallow.shape[-2:], mode="bilinear", align_corners=False
        )
        detail = shallow * torch.sigmoid(self.detail_gate(deep))
        edge = torch.sigmoid(self.boundary_gate(boundary))
        if self.config.mode in ("no_constraint", "no_boundary"):
            semantic = self.semantic(deep)
        else:
            semantic = self.semantic(
                constrained_pool(deep, probability, edge, self.config.affinity_strength)
            )
        gate = torch.sigmoid(self.competition(torch.cat((detail, semantic, boundary), 1)))
        if self.config.mode == "equal_fusion":
            gate = torch.full_like(gate, 0.5)
        fused = gate * detail + (1 - gate) * semantic
        return fused, gate


class BoundaryContrastHead(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.config = c = config.validate()
        w = c.width
        self.projections = nn.ModuleList(nn.Conv2d(ch, w, 1) for ch in channels)
        self.coarse = nn.Conv2d(w, 1, 1)
        self.boundary = InnerOuterBoundary(c)
        self.fusions = nn.ModuleList(BoundaryCrossScale(c) for _ in range(len(channels) - 1))
        if c.architecture == "custom_decoder":
            self.decoder = nn.Sequential(
                nn.Conv2d(w * len(channels), w, 3, padding=1),
                nn.GroupNorm(1, w),
                nn.GELU(),
                nn.Conv2d(w, 1, 1),
            )
        else:
            self.output_projections = nn.ModuleList(nn.Conv2d(w, ch, 1) for ch in channels)
            # Only gates are zero. Zeroing gates AND projections would stall the
            # final-loss gradient path. Auxiliary coarse supervision starts at step 1.
            self.alpha = nn.Parameter(torch.zeros(len(channels)))

    def forward(self, pyramid, output_size):
        levels = [layer(feature.float()) for layer, feature in zip(self.projections, pyramid)]
        coarse = self.coarse(levels[-1])
        probability = F.interpolate(
            coarse, levels[0].shape[-2:], mode="bilinear", align_corners=False
        ).sigmoid()
        source = 0.5 * (
            levels[0]
            + F.interpolate(levels[1], levels[0].shape[-2:], mode="bilinear", align_corners=False)
        )
        boundary, reliability, normal, boundary_mix = self.boundary(source, probability)
        refined, gates = [levels[-1]], []
        for module, level in zip(self.fusions, reversed(levels[:-1])):
            feature, gate = module(level, refined[-1], boundary, probability)
            refined.append(feature)
            gates.append(gate)
        refined.reverse()
        result = {
            "coarse_logits": coarse.float(),
            "probability": probability,
            "boundary": boundary,
            "reliability": reliability,
            "normal": normal,
            "boundary_mix": boundary_mix,
            "fusion_gates": gates,
        }
        if self.config.architecture == "custom_decoder":
            size = refined[0].shape[-2:]
            decoded = self.decoder(
                torch.cat(
                    [F.interpolate(x, size, mode="bilinear", align_corners=False) for x in refined],
                    1,
                )
            )
            result["logits"] = F.interpolate(
                decoded.float(), output_size, mode="bilinear", align_corners=False
            )
        else:
            # Each scale is projected separately to its ORIGINAL decoder-interface
            # channel count and grid. Never mix the three original bypass tensors.
            result["adapted_pyramid"] = [
                original + self.alpha[i] * projection(proposal)
                for i, (original, projection, proposal) in enumerate(
                    zip(pyramid, self.output_projections, refined)
                )
            ]
            result["feature_gates"] = self.alpha
        return result


class BoundaryContrastSAM2(nn.Module):
    def __init__(self, baseline, config):
        super().__init__()
        self.baseline = baseline.requires_grad_(False)
        self.image_size, self.config = baseline.image_size, config.validate()
        decoder = baseline.sam.sam_mask_decoder
        channels = [
            decoder.conv_s0.out_channels,
            decoder.conv_s1.out_channels,
            baseline.sam.sam_prompt_encoder.embed_dim,
        ]
        self.head = BoundaryContrastHead(channels, config)
        self.reference_decoder = (
            deepcopy(decoder).requires_grad_(False).eval() if config.train_decoder else None
        )
        if config.train_decoder:
            decoder.requires_grad_(True)
            # These projections run inside frozen forward_image, not final decode.
            decoder.conv_s0.requires_grad_(False)
            decoder.conv_s1.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        if self.config.train_decoder:
            self.baseline.sam.sam_mask_decoder.train(mode)
            self.reference_decoder.eval()
        return self

    def forward_details(self, images, include_baseline=True):
        base, sam = self.baseline, self.baseline.sam
        if images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Expected square SAM2 input at configured image_size")
        with torch.no_grad():
            pyramid = sam.forward_image((images - base.mean) / base.std)["backbone_fpn"][
                -sam.num_feature_levels :
            ]
            embed = pyramid[-1]
            if sam.directly_add_no_mem_embed:
                embed = embed + sam.no_mem_embed.reshape(1, -1, 1, 1)
            pyramid = [*pyramid[:-1], embed]
            reference = None
            if include_baseline:
                _, dense = sam.sam_prompt_encoder(points=None, boxes=None, masks=None)
                reference_decoder = (
                    self.reference_decoder
                    if self.reference_decoder is not None
                    else sam.sam_mask_decoder
                )
                masks, _, _, _ = reference_decoder(
                    image_embeddings=embed,
                    image_pe=sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=base.task_tokens.expand(len(images), -1, -1),
                    dense_prompt_embeddings=dense.expand(len(images), -1, -1, -1),
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=pyramid[:-1],
                )
                reference = F.interpolate(
                    masks.float(), images.shape[-2:], mode="bilinear", align_corners=False
                )
        result = self.head(pyramid, images.shape[-2:])
        if self.config.architecture == "baseline_adapter":
            features = result["adapted_pyramid"]
            # Frozen weights still differentiate with respect to adapted features.
            # Original task tokens, no-mask embedding and positional encoding stay.
            _, dense = sam.sam_prompt_encoder(points=None, boxes=None, masks=None)
            masks, _, _, _ = sam.sam_mask_decoder(
                image_embeddings=features[-1],
                image_pe=sam.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=base.task_tokens.expand(len(images), -1, -1),
                dense_prompt_embeddings=dense.expand(len(images), -1, -1, -1),
                multimask_output=False,
                repeat_image=False,
                high_res_features=features[:-1],
            )
            result["logits"] = F.interpolate(
                masks.float(), images.shape[-2:], mode="bilinear", align_corners=False
            )
        if reference is not None:
            result["baseline_logits"] = reference
        if not torch.isfinite(result["logits"]).all():
            raise FloatingPointError("Nonfinite boundary-contrast logits")
        return result

    def forward(self, images):
        return self.forward_details(images, include_baseline=False)["logits"]


def boundary_contrast_loss(result, target, valid, config):
    main = segmentation_loss(result["logits"], target, valid)
    # Supervise low-res coarse predictions at the target grid; do not silently
    # erase small foregrounds by nearest-neighbor downsampling their labels.
    coarse = F.interpolate(
        result["coarse_logits"], target.shape[-2:], mode="bilinear", align_corners=False
    )
    auxiliary = segmentation_loss(coarse, target, valid)
    return {
        "total": main + config.coarse_weight * auxiliary,
        "segmentation": main,
        "coarse": auxiliary,
    }
