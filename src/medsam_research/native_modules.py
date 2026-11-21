"""Experimental SCDS, SBCA and RCSR. No claims of established novelty or benefit."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GatedFFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 4, 1)
        self.spatial = nn.Conv2d(channels * 2, channels * 2, 3, padding=1,
                                 groups=channels * 2)
        self.project = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        a, b = self.expand(x).chunk(2, dim=1)
        return x + self.project(F.gelu(self.spatial(a)) * b)


class SCDS(nn.Module):
    """Refine one native-resolution shallow feature using two deep reference grids."""

    def __init__(self, channels, deep_channels=(576, 1152), width=64, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        self.local = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GroupNorm(1, width))
        self.dw3 = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.dw5 = nn.Conv2d(width, width, 5, padding=2, groups=width)
        self.scale_gate = nn.Conv2d(width, 2, 1)
        self.deep = nn.ModuleList(nn.Conv2d(c, width, 1) for c in deep_channels)
        self.position = nn.Linear(2, width)
        self.level_embedding = nn.Parameter(torch.randn(2, 1, width) * .02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.selection = nn.Sequential(nn.Conv2d(width * 3, width, 1), nn.GELU(),
                                       nn.Conv2d(width, width, 1), nn.Sigmoid())
        self.ffn = GatedFFN(width)
        self.output = nn.Conv2d(width, channels, 1)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, shallow, deep_features):
        z = self.local(shallow)
        weights = self.scale_gate(z).softmax(1)
        detail = weights[:, :1] * self.dw3(z) + weights[:, 1:] * self.dw5(z)
        references = []
        for level, (projection, feature) in enumerate(zip(self.deep, deep_features)):
            size = (min(self.pool_size, feature.shape[-2]),
                    min(self.pool_size, feature.shape[-1]))
            ref = F.adaptive_avg_pool2d(projection(feature), size)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, size[0], device=z.device),
                                    torch.linspace(-1, 1, size[1], device=z.device),
                                    indexing="ij")
            pos = self.position(torch.stack((xx, yy), -1).reshape(-1, 2).to(z.dtype))
            references.append(ref.flatten(2).transpose(1, 2) + pos + self.level_embedding[level])
        ref = self.norm(torch.cat(references, dim=1))
        query = self.norm(detail.flatten(2).transpose(1, 2))
        context = self.attention(query, ref, ref, need_weights=False)[0]
        context = context.transpose(1, 2).reshape_as(detail)
        gate = self.selection(torch.cat((detail, context, (detail - context).abs()), 1))
        refined = self.ffn(z + gate * detail + (1 - gate) * context)
        return shallow + self.gain.tanh() * self.output(refined)


def hierarchical_attention(query, key, value, height, width, window):
    """B,H,1,D query; B,H,N,D keys. Return evidence and normalized pixel mass.

    Window weights use MEAN logits, not logsumexp (which would reproduce global
    softmax). Padding has zero mass in both the local and window distributions.
    """
    batch, heads, count, dim = key.shape
    if count != height * width:
        raise ValueError("Spatial grid does not match key count")
    window = min(window, max(height, width))
    hp = math.ceil(height / window) * window
    wp = math.ceil(width / window) * window
    logits = (query @ key.transpose(-1, -2)).squeeze(-2) / math.sqrt(dim)
    logits = F.pad(logits.reshape(batch, heads, height, width),
                   (0, wp - width, 0, hp - height))
    valid = F.pad(torch.ones(height, width, device=key.device, dtype=torch.bool),
                  (0, wp - width, 0, hp - height))

    def partition(x):
        return x.reshape(*x.shape[:-2], hp // window, window, wp // window, window).transpose(
            -3, -2
        ).reshape(*x.shape[:-2], -1, window * window)

    logits, valid = partition(logits), partition(valid)
    counts = valid.sum(-1).clamp_min(1)
    score = logits.masked_fill(~valid, 0).sum(-1) / counts
    window_weights = score.softmax(-1)
    pixel_weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)
    mass = window_weights[..., None] * pixel_weights
    mass = mass.reshape(batch, heads, hp // window, wp // window, window, window)
    mass = mass.transpose(-3, -2).reshape(batch, heads, hp, wp)
    mass = mass[:, :, :height, :width].reshape(batch, heads, count)
    return (mass.unsqueeze(-2) @ value).squeeze(-2), mass


class SBCA(nn.Module):
    """Wrap ONE native attention; alter only the designated single-mask query.

    Other output/task tokens retain their native attention. Background reference
    is an internal learned query, not an extra prompt or ground-truth input.
    """

    def __init__(self, native_attention, mask_query_index, windows=(2, 4, 8)):
        super().__init__()
        self.base = native_attention
        self.mask_query_index = mask_query_index
        self.windows = tuple(windows)
        channels = native_attention.q_proj.in_features
        self.background_query = nn.Parameter(torch.randn(1, 1, channels) * .02)
        self.scale_gate = nn.Linear(channels, len(windows))
        self.contrast_gate = nn.Linear(native_attention.out_proj.in_features * 2, 1)
        self.gain = nn.Parameter(torch.zeros(()))
        self.last_evidence = None

    def forward(self, q, k, v):
        native = self.base(q, k, v)
        size = math.isqrt(k.shape[1])
        if size * size != k.shape[1]:
            raise ValueError("SBCA currently supports square embedding grids only")
        index = self.mask_query_index
        target_query = q[:, index:index + 1]
        if target_query.shape[1] != 1:
            raise ValueError("Mask-token index outside query sequence")
        heads = self.base.num_heads
        key = self.base._separate_heads(self.base.k_proj(k), heads)
        value = self.base._separate_heads(self.base.v_proj(v), heads)
        tq = self.base._separate_heads(self.base.q_proj(target_query), heads)
        bq = self.base._separate_heads(self.base.q_proj(
            self.background_query.expand(q.shape[0], -1, -1)), heads)
        scale_weights = self.scale_gate(target_query[:, 0]).softmax(-1)
        foreground, background, fg_mass, bg_mass = [], [], [], []
        for window in self.windows:
            ft, fm = hierarchical_attention(tq, key, value, size, size, window)
            bt, bm = hierarchical_attention(bq, key, value, size, size, window)
            foreground.append(ft)
            background.append(bt)
            fg_mass.append(fm)
            bg_mass.append(bm)

        def combine(items):
            stacked = torch.stack(items, 1)
            return (stacked * scale_weights[:, :, None, None]).sum(1)

        fg, bg = combine(foreground).flatten(1), combine(background).flatten(1)
        lam = self.contrast_gate(torch.cat((fg, bg), -1)).sigmoid()
        proposed = self.base.out_proj(fg - lam * bg)
        delta = proposed - native[:, index]
        selector = torch.zeros_like(native)
        selector[:, index] = delta
        # Auxiliary BCE supervises FG/BG relative spatial evidence, not mask-token naming.
        fm, bm = combine(fg_mass).mean(1), combine(bg_mass).mean(1)
        self.last_evidence = (fm.clamp_min(1e-8).log() - bm.clamp_min(1e-8).log()).reshape(
            q.shape[0], 1, size, size
        )
        return native + self.gain.tanh() * selector


class RCSR(nn.Module):
    """Region-conditioned resampling after the native first transposed convolution.

    Preserve native upsampling weights. Refine its output before high-res addition.
    Offsets are in output-grid pixels; magnitude bounded to two pixels per axis.
    """

    def __init__(self, channels=64, token_dim=256):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(token_dim, channels), nn.GELU())
        self.local = nn.Conv2d(channels, channels, 1)
        self.predict = nn.Sequential(nn.Conv2d(channels * 3, channels, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(channels, 27, 1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, x, high_res, region_token):
        if x.shape != high_res.shape:
            raise ValueError("RCSR expects matching projected high-resolution features")
        batch, channels, height, width = x.shape
        condition = self.condition(region_token)[:, :, None, None].expand_as(x)
        controls = self.predict(torch.cat((x, self.local(high_res), condition), 1))
        offsets = 2 * controls[:, :18].reshape(batch, 9, 2, height, width).tanh()
        weights = controls[:, 18:].softmax(1)
        yy, xx = torch.meshgrid(torch.arange(height, device=x.device, dtype=x.dtype),
                                torch.arange(width, device=x.device, dtype=x.dtype),
                                indexing="ij")
        locations = x.new_tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        gx = xx + locations[:, 0, None, None] + offsets[:, :, 0]
        gy = yy + locations[:, 1, None, None] + offsets[:, :, 1]
        grid = torch.stack((2 * (gx + .5) / width - 1, 2 * (gy + .5) / height - 1), -1)
        sampled = F.grid_sample(x[:, None].expand(-1, 9, -1, -1, -1).reshape(
            batch * 9, channels, height, width), grid.reshape(batch * 9, height, width, 2),
            mode="bilinear", padding_mode="border", align_corners=False)
        reconstructed = (sampled.reshape(batch, 9, channels, height, width)
                         * weights[:, :, None]).sum(1)
        return x + self.gain.tanh() * (reconstructed - x)
