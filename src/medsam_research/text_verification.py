"""Text localization and conditional recovery/suppression; experimental, not IPP/BID."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import segmentation_loss

REPAIR_MODES = ('full', 'no-intervention', 'no-direction', 'residual', 'balanced-residual')


class TextLocalizer(nn.Module):
    def __init__(self, channels, text_dim=512, width=16):
        super().__init__()
        if len(channels) != 3:
            raise ValueError('Three SAM2 pyramid levels required')
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.text_projection = nn.Linear(text_dim, width, bias=False)
        self.map_projections = nn.ModuleList(nn.Conv2d(1, width, 1, bias=False) for _ in channels)
        self.fusions = nn.ModuleList(nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                                  nn.GELU()) for _ in range(3))
        self.output = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pyramid, baseline, similarity, text):
        condition = self.text_projection(text.float())[:, :, None, None]
        x = None
        for level in (2, 1, 0):
            feature = pyramid[level].float()
            current = self.projections[level](feature)
            response = F.interpolate(similarity.float(), feature.shape[-2:], mode='bilinear',
                                     align_corners=False)
            current = current + self.map_projections[level](response) + condition
            if x is not None:
                current = current+F.interpolate(x, feature.shape[-2:], mode='bilinear',
                                                 align_corners=False)
            x = self.fusions[level](current)
        return baseline.float()+F.interpolate(self.output(x), baseline.shape[-2:], mode='bilinear',
                                               align_corners=False)


def entropy(logits):
    p = logits.sigmoid().clamp(1e-6, 1-1e-6)
    return -(p*p.log()+(1-p)*(1-p).log())


class DirectionalRepair(nn.Module):
    def __init__(self, channels, width=16, threshold=.5, mode='full'):
        super().__init__()
        if mode not in REPAIR_MODES or not 0 < threshold < 1:
            raise ValueError('Invalid repair mode or threshold')
        self.threshold, self.mode = threshold, mode
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.fusions = nn.ModuleList(nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                                  nn.GELU()) for _ in range(2))
        self.evidence = nn.Sequential(nn.Conv2d(width+5, width, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pyramid, visual, prior):
        x = self.projections[2](pyramid[2].float())
        for fusion, level in zip(self.fusions, (1, 0)):
            x = fusion(F.interpolate(x, pyramid[level].shape[-2:], mode='bilinear',
                                      align_corners=False)+self.projections[level](pyramid[level].float()))
        p, v = prior.float().sigmoid(), visual.float().sigmoid()
        fields = torch.cat((p, v, p-v, entropy(prior.float()), entropy(visual.float())), 1)
        fields = F.interpolate(fields, x.shape[-2:], mode='bilinear', align_corners=False)
        delta = F.interpolate(self.output(self.evidence(torch.cat((x, fields), 1))),
                              prior.shape[-2:], mode='bilinear', align_corners=False)
        dr, ds = delta[:, :1], delta[:, 1:]
        domain = prior.detach() >= math.log(self.threshold/(1-self.threshold))
        recovery, suppression = prior.float()+dr, -prior.float()+ds
        final = (prior.float()+dr-ds if self.mode in ('residual', 'balanced-residual')
                 else torch.where(domain, prior.float()-ds, prior.float()+dr))
        return {'logits': final, 'recovery_logits': recovery,
                'suppression_logits': suppression, 'domain': domain}


def _masked_bce(logits, target, mask):
    mask = mask.float()
    return (F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
            * mask).sum()/mask.sum().clamp_min(1)


def direction_loss(result, target, valid):
    domain = result['domain']
    return (_masked_bce(result['recovery_logits'], target, valid*(~domain))
            + _masked_bce(result['suppression_logits'], 1-target, valid*domain))


def intervene_prior(prior, generator=None):
    """Alter a random rectangle per image; never inspect a target or visual prediction."""
    altered = prior.detach().clone()
    height, width = prior.shape[-2:]
    for i in range(len(prior)):
        h = int(torch.randint(max(1, height//8), max(2, height//3)+1, (), generator=generator))
        w = int(torch.randint(max(1, width//8), max(2, width//3)+1, (), generator=generator))
        h, w = min(h, height), min(w, width)
        y = int(torch.randint(0, height-h+1, (), generator=generator))
        x = int(torch.randint(0, width-w+1, (), generator=generator))
        sign = 1 if int(torch.randint(0, 2, (), generator=generator)) else -1
        altered[i, :, y:y+h, x:x+w] = sign*6.
    return altered


def repair_objective(head, pyramid, visual, prior, target, valid, mode,
                     direction_weight=.5, consistency_weight=.1, generator=None):
    first = head(pyramid, visual, prior)
    loss = segmentation_loss(first['logits'], target, valid)
    directional = mode in ('full', 'no-intervention', 'balanced-residual')
    def auxiliary(result):
        if mode == 'balanced-residual':
            domain = result['domain']
            return (_masked_bce(result['logits'], target, valid*(~domain))
                    + _masked_bce(result['logits'], target, valid*domain))
        return direction_loss(result, target, valid)
    if directional:
        loss = loss+direction_weight*auxiliary(first)
    if mode != 'no-intervention':
        altered = intervene_prior(prior, generator)
        second = head(pyramid, visual, altered)  # recomputes the operation domains
        other = segmentation_loss(second['logits'], target, valid)
        if directional:
            other = other+direction_weight*auxiliary(second)
        consistency = ((first['logits'].sigmoid()-second['logits'].sigmoid()).abs()*valid).sum()
        consistency = consistency/valid.sum().clamp_min(1)
        # Average views, so full mode does not simply double the supervised loss scale.
        loss = .5*(loss+other)+consistency_weight*consistency
    return loss


def complementarity(text_prediction, visual_prediction, target, valid):
    """Measured against annotations, not proof of independent or causal evidence."""
    v = valid.astype(bool)
    p, q, y = text_prediction & v, visual_prediction & v, target.astype(bool) & v
    fn, fp = y & ~p, p & ~y
    fn_count, fp_count = int(fn.sum()), int(fp.sum())
    recovered, excluded = int((fn & q).sum()), int((fp & ~q).sum())
    return {'text_fn_pixels': fn_count, 'text_fp_pixels': fp_count,
            'visual_recovers_text_fn_pixels': recovered,
            'visual_rejects_text_fp_pixels': excluded,
            'visual_recovery_fraction': recovered/fn_count if fn_count else None,
            'visual_rejection_fraction': excluded/fp_count if fp_count else None}
