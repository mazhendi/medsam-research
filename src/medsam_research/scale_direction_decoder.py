"""Proposed SDS / ADF / RTR decoder; final-head-only inference, no GT inputs."""
import torch
from torch import nn
from torch.nn import functional as F

from .subband_decoder import resize


class ChannelMLP(nn.Sequential):
    def __init__(self, incoming, width):
        super().__init__(nn.Conv2d(incoming, width*2, 1), nn.GELU(),
                         nn.Conv2d(width*2, width, 1))


class SDS(nn.Module):
    """Spatial softmax selects five scale/direction experts; preserves X."""
    def __init__(self, width):
        super().__init__()
        kernels = ((3, 3), (5, 5), (7, 7), (1, 7), (7, 1))
        self.branches = nn.ModuleList(nn.Conv2d(width, width, kernel,
                                              padding=(kernel[0]//2, kernel[1]//2),
                                              groups=width) for kernel in kernels)
        self.weight_generator = nn.Sequential(nn.Conv2d(width*2, width, 1), nn.GELU(),
                                              nn.Conv2d(width, 5, 1))
        self.fuse = ChannelMLP(width*2, width)

    def spatial_weights(self, x, semantic):
        return self.weight_generator(torch.cat((x, resize(semantic, x)), 1)).softmax(1)

    def forward(self, x, semantic):
        weights = self.spatial_weights(x, semantic)
        selected = sum(branch(x)*weights[:, i:i+1] for i, branch in enumerate(self.branches))
        return self.fuse(torch.cat((x, selected), 1))


class ADF(nn.Module):
    """Agreement and disagreement are complementary descriptors, not error labels."""
    def __init__(self, width):
        super().__init__()
        self.fine_projection = nn.Conv2d(width, width, 1)
        self.deep_projection = nn.Conv2d(width, width, 1)
        self.agreement = nn.Sequential(nn.Conv2d(width*2, width, 3, padding=1), nn.GELU())
        self.disagreement = nn.Sequential(
            nn.Conv2d(width*2, width*2, 3, padding=1, groups=width*2), nn.GELU(),
            ChannelMLP(width*2, width))
        self.fuse = nn.Sequential(nn.Conv2d(width*4, width, 1),
                                  nn.Conv2d(width, width, 3, padding=1, groups=width),
                                  ChannelMLP(width, width))

    @staticmethod
    def relation_weights(x, y):
        cosine = (F.normalize(x, dim=1, eps=1e-6)*F.normalize(y, dim=1, eps=1e-6)).sum(1, True)
        agreement = ((1+cosine)/2).clamp(0, 1)
        return agreement, 1-agreement

    def forward(self, fine, deep):
        x, y = self.fine_projection(fine), self.deep_projection(resize(deep, fine))
        a, d = self.relation_weights(x, y)
        agree = self.agreement(torch.cat((x*y, x+y), 1))
        difference = x-y
        disagree = self.disagreement(torch.cat((difference, difference.abs()), 1))
        return self.fuse(torch.cat((x, y, a*agree, d*disagree), 1))


class LocalFusion(nn.Module):
    """ADF-off control with same input/output interface, not parameter-matched."""
    def __init__(self, width):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(width*2, width, 3, padding=1), nn.GELU(),
                                  ChannelMLP(width, width))

    def forward(self, fine, deep):
        return self.fuse(torch.cat((fine, resize(deep, fine)), 1))


def probability_geometry(probability, tau=.02, eps=1e-4):
    """Central differences, replicated borders; zero flat-field gate and finite gradients."""
    padded = F.pad(probability, (1, 1, 1, 1), mode="replicate")
    dx = (padded[..., 1:-1, 2:]-padded[..., 1:-1, :-2])/2
    dy = (padded[..., 2:, 1:-1]-padded[..., :-2, 1:-1])/2
    energy = dx.square()+dy.square()
    stabilized = (energy+eps**2).sqrt()
    normal = torch.cat((dx, dy), 1)/stabilized
    # Equivalent to sqrt(energy+eps²)-eps, without cancellation on a flat field.
    magnitude = energy/(stabilized+eps)
    gate = magnitude/(magnitude+tau)
    return normal, gate


def sample_along_normal(feature, normal, distance):
    batch, _, height, width = feature.shape
    if normal.shape != (batch, 2, height, width):
        raise ValueError("normal must match the feature grid and have two channels")
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=feature.device, dtype=feature.dtype)+.5)*2/height-1,
        (torch.arange(width, device=feature.device, dtype=feature.dtype)+.5)*2/width-1,
        indexing="ij")
    base = torch.stack((xx, yy), -1)[None]
    displacement = normal.permute(0, 2, 3, 1)*feature.new_tensor([2/width, 2/height])*distance
    return F.grid_sample(feature, base+displacement, mode="bilinear",
                         padding_mode="border", align_corners=False)


class RTR(nn.Module):
    def __init__(self, width, tau=.02):
        super().__init__()
        if tau <= 0:
            raise ValueError("RTR tau must be positive")
        self.tau = tau
        self.contrast_projection = nn.Conv2d(width*5, width, 1)
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.fuse = ChannelMLP(width*3, width)

    def forward(self, region, fine, probability):
        if region.shape != fine.shape or probability.shape != (fine.shape[0], 1, *fine.shape[-2:]):
            raise ValueError("RTR inputs must share a grid; region/fine must share channels")
        normal, gate = probability_geometry(probability, self.tau)
        contrasts, means = [], []
        for distance in (1., 2.):
            plus = sample_along_normal(fine, normal, distance)
            minus = sample_along_normal(fine, normal, -distance)
            contrasts.append(plus-minus)
            means.append((plus+minus)/2)
        descriptor = torch.cat((fine, *contrasts, *means), 1)
        contrast = self.contrast_projection(descriptor)
        mixed = gate*contrast+(1-gate)*self.local(fine)
        return self.fuse(torch.cat((region, fine, mixed), 1))


class ScaleDirectionPVT(nn.Module):
    def __init__(self, backbone, width=64, sds=True, adf=True, rtr=True, tau=.02):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width < 8:
            raise ValueError("width must be an integer >=8")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.backbone = backbone
        self.project = nn.ModuleList(nn.Conv2d(c, width, 1) for c in (64, 128, 320, 512))
        self.sds = nn.ModuleList(SDS(width) for _ in range(2)) if sds else None
        self.coarse_enhance = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width),
                          nn.Conv2d(width, width, 1)) for _ in range(2))
        self.initialize = ChannelMLP(width, width)
        self.fuse = nn.ModuleList((ADF if adf else LocalFusion)(width) for _ in range(3))
        self.coarse_head = nn.Conv2d(width, 1, 1)
        self.aux_heads = nn.ModuleList(nn.Conv2d(width, 1, 1) for _ in range(2))
        self.rtr = RTR(width, tau) if rtr else None
        self.final_head = nn.Conv2d(width, 1, 1)

    def forward(self, image):
        projected = [p(f) for p, f in zip(self.project, self.backbone(image))]
        e3, semantic = [m(f) for m, f in zip(self.coarse_enhance, projected[2:])]
        e1, e2 = ([m(f, semantic) for m, f in zip(self.sds, projected[:2])]
                  if self.sds is not None else projected[:2])
        d4 = self.initialize(semantic)
        d3 = self.fuse[2](e3, d4)
        d2 = self.fuse[1](e2, d3)
        d1 = self.fuse[0](e1, d2)
        coarse_native = self.coarse_head(d1)
        refined = self.rtr(d1, e1, coarse_native.sigmoid()) if self.rtr is not None else d1

        def up(logits):
            return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)

        return dict(final=up(self.final_head(refined)), coarse=up(coarse_native),
                    aux2=up(self.aux_heads[0](d2)), aux3=up(self.aux_heads[1](d3)))


def scale_direction_loss(outputs, target, coarse_weight=.5, intermediate_weight=.25):
    from .sam2_unet_workflow import structure_loss

    return (structure_loss(outputs["final"], target)
            + coarse_weight*structure_loss(outputs["coarse"], target)
            + intermediate_weight*(structure_loss(outputs["aux2"], target)
                                   + structure_loss(outputs["aux3"], target)))
