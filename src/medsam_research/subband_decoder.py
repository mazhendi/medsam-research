"""Proposed replacement decoder; PVT encoder only, no original CIM/CFM/SAM.

SCSI: per-location four-band attention conditioned on deep semantics.
BCSC: parallel adjacent-scale pairs with two independent local samplers.
Region/boundary paths exchange features simultaneously at each decoding level.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(x, reference):
    return F.interpolate(x, reference.shape[-2:], mode="bilinear", align_corners=False)


def haar_split(x):
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar split requires even height and width")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    # Band axis: LL, LH, HL, HH; normalization gives an orthonormal transform.
    return torch.stack(((a+b+c+d)/2, (a-b+c-d)/2,
                        (a+b-c-d)/2, (a-b-c+d)/2), dim=1)


def haar_merge(bands):
    if bands.ndim != 5 or bands.shape[1] != 4:
        raise ValueError("Expected B,4,C,H,W bands")
    ll, lh, hl, hh = bands.unbind(1)
    a, b = (ll+lh+hl+hh)/2, (ll-lh+hl-hh)/2
    c, d = (ll+lh-hl-hh)/2, (ll-lh-hl+hh)/2
    # Interleave columns, then rows. No in-place writes into differentiable buffers.
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


class ConvMLP(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.GELU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, groups=outgoing),
                         nn.GELU(), nn.Conv2d(outgoing, outgoing, 1))


class SCSI(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.q, self.k, self.v = (nn.Linear(width, width) for _ in range(3))
        self.semantic_q = nn.Conv2d(width, width, 1)
        self.semantic_k = nn.Conv2d(width, width, 1)
        self.band_bias = nn.Parameter(torch.eye(4) * 2)
        self.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                 nn.Linear(2*width, width))
        self.scale = width ** -.5

    def forward(self, x, semantic):
        bands = haar_split(x)
        # B,H/2,W/2,4,C; attention is only 4x4, not global spatial attention.
        tokens = bands.permute(0, 3, 4, 1, 2)
        normalized = self.norm(tokens)
        context = F.interpolate(semantic, bands.shape[-2:], mode="bilinear", align_corners=False)
        sq = self.semantic_q(context).permute(0, 2, 3, 1).unsqueeze(-2)
        sk = self.semantic_k(context).permute(0, 2, 3, 1).unsqueeze(-2)
        q = self.q(normalized) * (1 + sq.tanh())
        k = self.k(normalized) * (1 + sk.tanh())
        weights = (q @ k.transpose(-1, -2) * self.scale + self.band_bias).softmax(-1)
        mixed = weights @ self.v(tokens)
        updated = self.mlp(mixed).permute(0, 3, 4, 1, 2)
        return haar_merge(updated)


class LocalCrossRead(nn.Module):
    """Queries on target grid; K,V sampled from source in source-pixel units."""
    def __init__(self, width, attention_dim=16, max_offset=2.):
        super().__init__()
        self.q = nn.Conv2d(width, attention_dim, 1)
        self.kv = nn.Conv2d(width, attention_dim*2, 1)
        self.offset = nn.Sequential(nn.Conv2d(width*2, width//2, 1), nn.GELU(),
                                    nn.Conv2d(width//2, 18, 1))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.fuse = ConvMLP(width+attention_dim, width)
        self.scale, self.max_offset = attention_dim ** -.5, max_offset
        points = [(x, y) for y in (-1., 0., 1.) for x in (-1., 0., 1.)]
        self.register_buffer("points", torch.tensor(points), persistent=False)

    def forward(self, target, source):
        batch, _, height, width = target.shape
        sh, sw = source.shape[-2:]
        offsets = self.offset(torch.cat((target, resize(source, target)), dim=1))
        offsets = offsets.reshape(batch, 9, 2, height, width).permute(0, 1, 3, 4, 2)
        offsets = offsets.tanh() * self.max_offset
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=target.device, dtype=target.dtype)+.5)*2/height-1,
            (torch.arange(width, device=target.device, dtype=target.dtype)+.5)*2/width-1,
            indexing="ij")
        base = torch.stack((xx, yy), dim=-1)[None]
        conversion = target.new_tensor([2/sw, 2/sh])
        kv = self.kv(source)
        sampled = []
        for index, point in enumerate(self.points):
            grid = base + (point + offsets[:, index]) * conversion
            sampled.append(F.grid_sample(kv, grid, mode="bilinear", padding_mode="border",
                                         align_corners=False))
        keys, values = torch.stack(sampled, dim=2).chunk(2, dim=1)
        weights = (self.q(target).unsqueeze(2) * keys).sum(1) * self.scale
        attended = (weights.softmax(1).unsqueeze(1) * values).sum(2)
        return self.fuse(torch.cat((target, attended), dim=1))


class BCSC(nn.Module):
    def __init__(self, width, attention_dim=16):
        super().__init__()
        self.fine_reads = nn.ModuleList(LocalCrossRead(width, attention_dim) for _ in range(3))
        self.coarse_reads = nn.ModuleList(LocalCrossRead(width, attention_dim) for _ in range(3))
        self.merge = nn.ModuleList(ConvMLP(width*n, width) for n in (1, 2, 2, 1))

    def forward(self, features):
        messages = [[] for _ in range(4)]
        # All pairs read original inputs, avoiding an undocumented serial update order.
        for index in range(3):
            fine, coarse = features[index:index+2]
            messages[index].append(self.fine_reads[index](fine, coarse))
            messages[index+1].append(self.coarse_reads[index](coarse, fine))
        return [merge(torch.cat(parts, dim=1)) for merge, parts in zip(self.merge, messages)]


class RegionBoundaryDecoder(nn.Module):
    def __init__(self, width, boundary=True):
        super().__init__()
        self.boundary_enabled = boundary
        self.region_start = ConvMLP(width, width)
        self.region_up = nn.ModuleList(ConvMLP(width*2, width) for _ in range(3))
        self.coarse_head, self.refined_head = nn.Conv2d(width, 1, 1), nn.Conv2d(width, 1, 1)
        if boundary:
            self.boundary_start = ConvMLP(width, width)
            self.boundary_up = nn.ModuleList(ConvMLP(width*2, width) for _ in range(3))
            self.region_exchange = nn.ModuleList(ConvMLP(width*2, width) for _ in range(3))
            self.boundary_exchange = nn.ModuleList(ConvMLP(width*2, width) for _ in range(3))
            self.edge_head = nn.Conv2d(width, 1, 1)

    def forward(self, features, output_size):
        region = self.region_start(features[3])
        boundary = self.boundary_start(features[3]) if self.boundary_enabled else None
        coarse = None
        for index in (2, 1, 0):
            feature = features[index]
            r = self.region_up[index](torch.cat((feature, resize(region, feature)), dim=1))
            if self.boundary_enabled:
                b = self.boundary_up[index](torch.cat((feature, resize(boundary, feature)), dim=1))
                # Simultaneous exchange, no multiplication by a coarse predicted mask.
                region = self.region_exchange[index](torch.cat((r, b), dim=1))
                boundary = self.boundary_exchange[index](torch.cat((b, r), dim=1))
            else:
                region = r
            if index == 1:
                coarse = self.coarse_head(region)
        def up(x):
            return F.interpolate(x, output_size, mode="bilinear", align_corners=False)
        result = {"coarse": up(coarse), "refined": up(self.refined_head(region))}
        if self.boundary_enabled:
            result["boundary"] = up(self.edge_head(boundary))
        return result


class SubbandPVT(nn.Module):
    def __init__(self, backbone, width=64, scsi=True, bcsc=True, boundary=True, attention_dim=16):
        super().__init__()
        if width < 16 or width % 8 or attention_dim < 1:
            raise ValueError("width must be a multiple of 8 >=16; attention_dim must be positive")
        self.backbone = backbone
        self.project = nn.ModuleList(nn.Conv2d(c, width, 1) for c in (64, 128, 320, 512))
        self.scsi = nn.ModuleList(SCSI(width) for _ in range(3)) if scsi else None
        self.bcsc = BCSC(width, attention_dim) if bcsc else None
        self.decoder = RegionBoundaryDecoder(width, boundary)

    def forward(self, x):
        features = [project(f) for project, f in zip(self.project, self.backbone(x))]
        if self.scsi is not None:
            features = [m(f, features[3]) for m, f in zip(self.scsi, features[:3])] + features[3:]
        if self.bcsc is not None:
            features = self.bcsc(features)
        return self.decoder(features, x.shape[-2:])


def boundary_target(mask):
    binary = (mask >= .5).to(mask.dtype)
    # 3x3 morphological gradient, derived solely from training GT.
    return F.max_pool2d(binary, 3, 1, 1) + F.max_pool2d(-binary, 3, 1, 1)


def decoder_loss(outputs, target, edge_weight=.2):
    from .sam2_unet_workflow import structure_loss

    loss = structure_loss(outputs["coarse"], target) + structure_loss(outputs["refined"], target)
    if edge_weight and "boundary" in outputs:
        edge = boundary_target(target)
        loss = loss + edge_weight * structure_loss(outputs["boundary"], edge)
    return loss
