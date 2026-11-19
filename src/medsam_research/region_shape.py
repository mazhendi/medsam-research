"""Region/implicit-contour co-inference prototype; no residual mask output.

The baseline supplies one frozen image encoding and initial evidence. Final
logits are decoded directly from joint dense/sparse prompts by SAM2. No labels,
SDFs, oracle hypothesis choices or uncertainty annotations enter forward().
"""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.nn import functional as F

from .metrics import segmentation_loss

MODES = ('full', 'no_exchange', 'uniform_competition', 'region_only')


@dataclass
class RegionShapeConfig:
    width: int = 64
    heads: int = 4
    hypotheses: int = 4
    rounds: int = 2
    grid_size: int = 64
    temperature: float = .2
    field_clip: float = 8.
    region_weight: float = .1
    field_weight: float = .1
    consensus_weight: float = .1
    diversity_weight: float = .01
    diversity_margin: float = .05
    train_decoder: bool = False
    mode: str = 'full'
    routing: str = 'global'
    window_size: int = 8
    local_strength: float = .5

    def validate(self):
        for name in ('width', 'heads', 'hypotheses', 'rounds', 'grid_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.width % self.heads or self.grid_size < 4:
            raise ValueError('width must divide into heads; grid_size must be >=4')
        for name in ('temperature', 'field_clip'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('region_weight', 'field_weight', 'consensus_weight', 'diversity_weight',
                     'diversity_margin'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if type(self.train_decoder) is not bool or self.mode not in MODES:
            raise ValueError('Invalid train_decoder or mode')
        if self.routing not in ('global', 'window'):
            raise ValueError('routing must be global or window')
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError('window_size must be a positive integer')
        if self.routing == 'window' and self.grid_size % self.window_size:
            raise ValueError('grid_size must be divisible by window_size for window routing')
        if not np.isfinite(self.local_strength) or not 0 <= self.local_strength <= 1:
            raise ValueError('local_strength must be finite and in [0,1]')
        return self

    def to_dict(self):
        return asdict(self)


def load_mechanism_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    allowed = {f.name for f in fields(RegionShapeConfig)}
    if not isinstance(raw, dict) or set(raw)-allowed:
        raise ValueError('Unknown region-shape configuration fields')
    return RegionShapeConfig(**raw).validate()


def weighted_hypotheses(occupancy, weights):
    """Moments of either global BxK or spatial BxKxHxW probability weights."""
    if weights.ndim == 2:
        if weights.shape != occupancy.shape[:2]:
            raise ValueError('Global weight dimensions disagree with occupancies')
        weights = weights[:, :, None, None]
    elif weights.shape != occupancy.shape:
        raise ValueError('Spatial weight dimensions disagree with occupancies')
    mean = (weights*occupancy).sum(1, keepdim=True)
    variance = (weights*(occupancy-mean).square()).sum(1, keepdim=True)
    return mean, variance


class RegionShapeHead(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.config = c = config.validate()
        w = c.width
        self.projections = nn.ModuleList(nn.Conv2d(ch, w, 1) for ch in channels)
        self.fusion = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(1, w), nn.GELU())
        self.position = nn.Sequential(nn.Conv2d(2, w, 1), nn.GELU(), nn.Conv2d(w, w, 1))
        self.queries = nn.Parameter(torch.randn(c.hypotheses, w)*.02)
        self.memory = nn.Conv2d(w+1, w, 1)
        self.query_attention = nn.MultiheadAttention(w, c.heads, dropout=0., batch_first=True)
        self.query_norm = nn.LayerNorm(w)
        self.query_mlp = nn.Sequential(nn.Linear(w, 2*w), nn.GELU(), nn.Linear(2*w, w))
        self.query_norm2 = nn.LayerNorm(w)
        self.field_spatial = nn.Conv2d(w, w, 1)
        self.field_query = nn.Linear(w, w)
        self.field_bias = nn.Linear(w, 1)
        self.compatibility = nn.Sequential(nn.Linear(4*w+2, w), nn.GELU(), nn.Linear(w, 1))
        self.region_update = nn.Sequential(nn.Conv2d(w+3, w, 3, padding=1),
                                            nn.GroupNorm(1, w), nn.GELU(), nn.Conv2d(w, 1, 1))

    def competition(self, feature, probability, queries, occupancy, return_spatial=False):
        b, k, height, width = occupancy.shape
        flat, f = occupancy.flatten(2), feature.flatten(2)
        inner = torch.einsum('bkn,bcn->bkc', flat, f)/flat.sum(-1, keepdim=True).clamp_min(1e-6)
        outer = torch.einsum('bkn,bcn->bkc', 1-flat, f)/(1-flat).sum(-1, keepdim=True).clamp_min(1e-6)
        p = probability.float().clamp(1e-5, 1-1e-5)
        # Cross-entropy compatibility is differentiable in BOTH states. It is a
        # consistency signal, not a claim that the regional prediction is true.
        disagreement = -(occupancy*p.log()+(1-occupancy)*(1-p).log()).mean((-2, -1))
        area = occupancy.mean((-2, -1))
        descriptors = torch.cat((queries, inner, outer, inner-outer,
                                 disagreement[..., None], area[..., None]), -1)
        energy = disagreement+.1*self.compatibility(descriptors).squeeze(-1).float()
        weights = torch.softmax(-energy, dim=1)
        if self.config.mode == 'uniform_competition':
            weights = torch.ones_like(weights)/k
        spatial = weights[:, :, None, None].expand_as(occupancy)
        if (self.config.routing == 'window' and self.config.local_strength > 0
                and self.config.mode != 'uniform_competition'):
            spatial = self.window_routing(feature, probability, queries, occupancy, energy)
            mean, variance = weighted_hypotheses(occupancy, spatial)
        else:
            mean, variance = weighted_hypotheses(occupancy, weights)
        result = (weights, mean, variance, energy)
        return (*result, spatial) if return_spatial else result

    def window_routing(self, feature, probability, queries, occupancy, global_energy):
        """Share the original scoring MLP; only aggregation granularity changes.

        Convex energy blend anchors local choices to global compatibility. Routing
        is dense, soft and computed on windows, not sparse expert execution.
        """
        c = self.config
        b, k, height, width = occupancy.shape
        ws = c.window_size
        if height % ws or width % ws:
            raise ValueError('Occupancy grid must divide into routing windows')
        gh, gw = height//ws, width//ws

        def windows(tensor):
            channels = tensor.shape[1]
            return tensor.float().reshape(b, channels, gh, ws, gw, ws).permute(
                0, 2, 4, 1, 3, 5).reshape(b, gh*gw, channels, ws*ws)

        h, f = windows(occupancy), windows(feature)
        p = windows(probability).clamp(1e-5, 1-1e-5)
        inner = torch.einsum('bwkn,bwcn->bwkc', h, f)/h.sum(-1, keepdim=True).clamp_min(1e-6)
        outer = torch.einsum('bwkn,bwcn->bwkc', 1-h, f)/(1-h).sum(-1, keepdim=True).clamp_min(1e-6)
        disagreement = -(h*p.log()+(1-h)*(1-p).log()).mean(-1)
        area = h.mean(-1)
        q = queries[:, None].expand(-1, gh*gw, -1, -1)
        descriptor = torch.cat((q, inner, outer, inner-outer,
                                disagreement[..., None], area[..., None]), -1)
        local_energy = disagreement+.1*self.compatibility(descriptor).squeeze(-1).float()
        energy = ((1-c.local_strength)*global_energy[:, None]
                  +c.local_strength*local_energy)
        weights = energy.neg().softmax(-1).transpose(1, 2).reshape(b, k, gh, gw)
        weights = F.interpolate(weights, (height, width), mode='bilinear', align_corners=False)
        # Interpolation of probability simplices preserves positivity. Normalize
        # again only for roundoff; no test labels or best-candidate choices here.
        return weights/weights.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, pyramid, initial_logits):
        c = self.config
        size = (c.grid_size, c.grid_size)
        projected = [F.interpolate(layer(level.float()), size, mode='bilinear', align_corners=False)
                     for layer, level in zip(self.projections, pyramid)]
        feature = self.fusion(torch.stack(projected).mean(0))
        axis = torch.linspace(-1, 1, c.grid_size, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        feature = feature+.1*self.position(torch.stack((xx, yy))[None])
        region = F.interpolate(initial_logits.detach().float(), size, mode='bilinear', align_corners=False)
        queries = self.queries[None].expand(len(region), -1, -1)
        history = []
        for _ in range(c.rounds):
            probability = region.sigmoid()
            if c.mode not in ('no_exchange', 'region_only'):
                memory = self.memory(torch.cat((feature, probability), 1)).flatten(2).transpose(1, 2)
                update, _ = self.query_attention(queries, memory, memory, need_weights=False)
                queries = self.query_norm(queries+update)
                queries = self.query_norm2(queries+self.query_mlp(queries))
            field = torch.einsum('bkc,bchw->bkhw', self.field_query(queries),
                                  self.field_spatial(feature))/(c.width**.5)
            field = torch.tanh(field+self.field_bias(queries)[..., None])
            occupancy = torch.sigmoid(-field.float()/c.temperature)
            weights, mean, variance, energy, spatial = self.competition(
                feature, probability, queries, occupancy, return_spatial=True)
            if c.mode == 'region_only':
                mean, variance = probability, torch.zeros_like(probability)
            if c.mode != 'no_exchange':
                region = self.region_update(torch.cat((feature, probability, mean, variance), 1)).float()
            history.append({'region': region, 'field': field.float(), 'occupancy': occupancy,
                            'weights': weights, 'spatial_weights': spatial, 'mean': mean, 'variance': variance, 'energy': energy})
        # Recompute compatibility against the FINAL regional state (not stale p_t).
        probability = region.sigmoid()
        weights, mean, variance, energy, spatial = self.competition(
                feature, probability, queries, occupancy, return_spatial=True)
        if c.mode == 'region_only':
            mean, variance = probability, torch.zeros_like(probability)
        return {'region': region, 'probability': probability, 'queries': queries,
                'field': field.float(), 'occupancy': occupancy, 'weights': weights,
                'spatial_weights': spatial, 'mean': mean, 'variance': variance, 'energy': energy, 'history': history}


class RegionShapeSAM2(nn.Module):
    def __init__(self, baseline, config):
        super().__init__()
        self.config = c = config.validate()
        self.baseline = baseline
        baseline.requires_grad_(False)
        if c.train_decoder:
            baseline.sam.sam_mask_decoder.requires_grad_(True)
        self.image_size = baseline.image_size
        if c.grid_size > self.image_size:
            raise ValueError('Mechanism grid exceeds model input size')
        decoder = baseline.sam.sam_mask_decoder
        dim = baseline.sam.sam_prompt_encoder.embed_dim
        channels = [decoder.conv_s0.out_channels, decoder.conv_s1.out_channels, dim]
        self.head = RegionShapeHead(channels, c)
        self.dense_prompt = nn.Sequential(nn.Conv2d(3, c.width, 3, padding=1), nn.GELU(),
                                           nn.Conv2d(c.width, dim, 1))
        self.task_query = nn.Linear(dim, c.width, bias=False)
        self.sparse_prompt = nn.Linear(c.width, dim, bias=False)
        # Identity initialization preserves the trained baseline without adding
        # baseline logits to final logits. This is prompt initialization only.
        nn.init.zeros_(self.dense_prompt[-1].weight)
        with torch.no_grad():
            self.dense_prompt[-1].bias.copy_(baseline.sam.sam_prompt_encoder.no_mask_embed.weight[0])
        nn.init.zeros_(self.sparse_prompt.weight)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.config.train_decoder:
            self.baseline.train(mode)
        else:
            self.baseline.eval()
        return self

    def forward_details(self, images):
        # One image encoder call, one initial decoder call, one final decoder call.
        with torch.set_grad_enabled(self.training and self.config.train_decoder):
            initial, pyramid = self.baseline.forward_with_pyramid(images)
        state = self.head(pyramid, initial)
        cues = torch.cat((state['probability'], state['mean'], state['variance']), 1)
        dense = self.dense_prompt(cues)
        dense = F.interpolate(dense, pyramid[-1].shape[-2:], mode='bilinear', align_corners=False)
        task = self.baseline.task_tokens.expand(len(images), -1, -1)
        scores = torch.einsum('btc,bkc->btk', self.task_query(task), state['queries'])/(self.config.width**.5)
        scores = scores.float()+state['weights'].clamp_min(1e-6).log()[:, None]
        context = torch.einsum('btk,bkc->btc', scores.softmax(-1).to(state['queries'].dtype), state['queries'])
        if self.config.mode == 'region_only':
            context = torch.zeros_like(context)
        sparse = task+self.sparse_prompt(context)
        sam = self.baseline.sam
        masks, _, _, _ = sam.sam_mask_decoder(
            image_embeddings=pyramid[-1], image_pe=sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=pyramid[:-1])
        logits = F.interpolate(masks.float(), images.shape[-2:], mode='bilinear', align_corners=False)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite joint decoder output')
        return {'logits': logits, 'baseline_logits': initial.detach(), **state}

    def forward(self, images):
        return self.forward_details(images)['logits']


def coarse_targets(target, valid, size, field_clip):
    """Coverage-preserving auxiliary labels; SDF is training supervision ONLY."""
    if valid is None:
        valid = torch.ones_like(target)
    clean = target.float()*valid.float()
    coarse = F.adaptive_max_pool2d(clean, size)
    weight = F.adaptive_max_pool2d(valid.float(), size)
    arrays = (coarse.detach().cpu().numpy()[:, 0] > .5)
    sdf = []
    for mask in arrays:
        if not mask.any():
            field = np.ones_like(mask, dtype=np.float32)
        elif mask.all():
            field = -np.ones_like(mask, dtype=np.float32)
        else:
            field = (distance_transform_edt(~mask)-distance_transform_edt(mask))/field_clip
            field = np.clip(field, -1, 1).astype(np.float32)
        sdf.append(field)
    return coarse, weight, torch.as_tensor(np.stack(sdf), device=target.device)[:, None]


def region_shape_loss(result, target, valid, config):
    c = config
    if valid is None:
        valid = torch.ones_like(target)
    final = segmentation_loss(result['logits'], target, valid)
    size = result['region'].shape[-2:]
    coarse, weight, sdf = coarse_targets(target, valid, size, c.field_clip)
    if (weight.flatten(1).sum(-1) == 0).any():
        raise ValueError('Every sample needs valid pixels')
    den = weight.sum((-2, -1)).clamp_min(1e-6)
    region = torch.stack([segmentation_loss(s['region'], coarse, weight) for s in result['history']]).mean()
    zero = final*0
    field_loss = consensus_loss = diversity_loss = zero
    if c.mode != 'region_only':
        per_hypothesis = ((result['field']-sdf).abs()*weight).sum((-2, -1))/den
        field_loss = per_hypothesis.min(1).values.mean()
        consensus_loss = segmentation_loss(torch.logit(result['mean'].clamp(1e-5, 1-1e-5)), coarse, weight)
        occupancy = result['occupancy']
        if occupancy.shape[1] > 1:
            # Mild anti-collapse regularizer, not a claim that all alternative
            # fields are valid annotations. Restrict it to target boundary cells.
            band = (sdf.abs() <= .25).float()*weight
            pairs = []
            for i in range(occupancy.shape[1]):
                for j in range(i):
                    distance = ((occupancy[:, i:i+1]-occupancy[:, j:j+1]).abs()*band).sum((-2, -1))
                    count = band.sum((-2, -1))
                    distance = distance/count.clamp_min(1)
                    pairs.append((F.relu(c.diversity_margin-distance)*(count > 0)).mean())
            diversity_loss = torch.stack(pairs).mean()
    total = (final+c.region_weight*region+c.field_weight*field_loss
             +c.consensus_weight*consensus_loss+c.diversity_weight*diversity_loss)
    return {'total': total, 'segmentation': final, 'region': region, 'field': field_loss,
            'consensus': consensus_loss, 'diversity': diversity_loss}
