"""Stage-one shared detail/context correction; no learned action selection yet."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .data import audit_manifests, file_hash, image_tensor, load_fov, load_pair, read_manifest
from .diagnostics import error_overlay, vessel_metrics
from .engine import _restore, git_revision, set_seed, write_csv, write_json
from .metrics import binary_metrics, segmentation_loss


def center_boxes(width, height, grid=2):
    """Disjoint correction centers, covering every pixel exactly once."""
    xs = [round(i * width / grid) for i in range(grid + 1)]
    ys = [round(i * height / grid) for i in range(grid + 1)]
    if min(width, height) < grid or grid < 1:
        raise ValueError("Invalid center grid")
    return [(xs[x], ys[y], xs[x+1], ys[y+1]) for y in range(grid) for x in range(grid)]


def read_box(center, width, height, action):
    if action not in (0, 1):
        raise ValueError("Action must be detail=0 or context=1")
    if action == 0:
        return center
    x0, y0, x1, y1 = center
    dx, dy = (x1-x0)//2, (y1-y0)//2
    return max(0, x0-dx), max(0, y0-dy), min(width, x1+dx), min(height, y1+dy)


def roi(tensor, box, bounds, size):
    """Sample the same native-pixel rectangle from any feature/logit grid."""
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = bounds
    xs = (torch.arange(size, device=tensor.device) + .5) / size * (x1-x0) + x0
    ys = (torch.arange(size, device=tensor.device) + .5) / size * (y1-y0) + y0
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack((2*(xx-bx0)/(bx1-bx0)-1, 2*(yy-by0)/(by1-by0)-1), -1)[None]
    return F.grid_sample(tensor.float(), grid, mode='bilinear', padding_mode='border',
                         align_corners=False)


class CorrectionHead(nn.Module):
    def __init__(self, channels=256, hidden=32):
        super().__init__()
        self.projection = nn.Conv2d(channels, hidden, 1)
        self.body = nn.Sequential(nn.Conv2d(2*hidden+4, hidden, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(hidden, 1, 1))
        # Identity initial output, then learn residual corrections.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, global_feature, local_feature, global_logits, local_logits, action):
        marker = torch.full_like(global_logits, float(action))
        x = torch.cat((self.projection(global_feature), self.projection(local_feature),
                       global_logits, local_logits, local_logits-global_logits, marker), 1)
        return global_logits + self.body(x)


@torch.no_grad()
def encode(model, image, device):
    return model.forward_with_features(image_tensor(image, model.image_size)[None].to(device))


def region_inputs(model, image, global_pair, center, action, device, size):
    bounds = (0, 0, image.width, image.height)
    box = read_box(center, image.width, image.height, action)
    local_logits, local_features = encode(model, image.crop(box), device)
    global_logits, global_features = global_pair
    return (roi(global_features, center, bounds, size), roi(local_features, center, box, size),
            roi(global_logits, center, bounds, size), roi(local_logits, center, box, size))


def score(logits, target, valid, cfg):
    prediction = logits.sigmoid().ge(cfg.training.threshold).cpu().numpy()
    return {**binary_metrics(prediction, target, cfg.training.boundary_tolerance, valid),
            **vessel_metrics(prediction, target, valid)}, prediction


@torch.no_grad()
def validate(model, head, records, cfg, device, grid, size, out=None):
    head.eval()
    rows, region_rows = [], []
    for index, record in enumerate(records):
        image, target = load_pair(record, cfg.data.mask_threshold)
        valid = load_fov(record, target.shape)
        global_pair = encode(model, image, device)
        baseline = F.interpolate(global_pair[0].float(), target.shape, mode='bilinear',
                                 align_corners=False)[0, 0]
        predictions = {'global': baseline}
        for action, name in ((0, 'detail'), (1, 'context')):
            assembled = baseline.clone()
            for center in center_boxes(image.width, image.height, grid):
                x0, y0, x1, y1 = center
                inputs = region_inputs(model, image, global_pair, center, action, device, size)
                corrected = head(*inputs, action)
                # Stitch a correction residual onto the exact native baseline, preserving
                # the baseline when the head is zero (no down/up-sampling degradation).
                residual = corrected - inputs[2]
                delta = F.interpolate(residual, (y1-y0, x1-x0), mode='bilinear',
                                      align_corners=False)[0, 0]
                assembled[y0:y1, x0:x1] += delta
                if out is not None and valid[y0:y1, x0:x1].any():
                    region_target = target[y0:y1, x0:x1]
                    region_valid = valid[y0:y1, x0:x1]
                    before, _ = score(baseline[y0:y1, x0:x1], region_target, region_valid, cfg)
                    after, _ = score(assembled[y0:y1, x0:x1], region_target, region_valid, cfg)
                    region_rows.append({"id": record.sample_id, "group": record.group,
                                        "action": name, "x0": x0, "y0": y0,
                                        "x1": x1, "y1": y1,
                                        "baseline_dice": before["dice"],
                                        "corrected_dice": after["dice"],
                                        "delta_dice": after["dice"]-before["dice"],
                                        "delta_precision": after["precision"]-before["precision"],
                                        "delta_recall": after["recall"]-before["recall"]})
            predictions[name] = assembled
        for mode, logits in predictions.items():
            metrics, prediction = score(logits, target, valid, cfg)
            rows.append({'id': record.sample_id, 'dataset': record.dataset,
                         'group': record.group, 'mode': mode, **metrics})
            if out is not None:
                prefix = f'{index:06d}_{mode}'
                error_overlay(image, prediction, target, valid).save(out / f'{prefix}_errors.png')
                Image.fromarray((prediction & valid.astype(bool)).astype(np.uint8)*255).save(
                    out / f'{prefix}_mask.png')
    summary = {}
    for mode in ('global', 'detail', 'context'):
        selected = [r for r in rows if r['mode'] == mode]
        summary[mode] = {k: float(np.mean([r[k] for r in selected]))
                         for k in ('dice', 'iou', 'boundary_f1', 'precision', 'recall', 'cldice')}
    if out is not None and region_rows:
        write_csv(out / "per_region.csv", region_rows)
    return summary, rows


def train_correction(cfg, checkpoint, output, epochs=20, grid=2, size=128, lr=1e-3,
                     preservation=0.1):
    if epochs < 1 or grid < 1 or size < 8 or lr <= 0 or preservation < 0:
        raise ValueError('Invalid correction training settings')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    set_seed(cfg.training.seed)
    audit = audit_manifests([cfg.data.train_manifest, cfg.data.val_manifest],
                           cfg.data.mask_threshold)
    model, device = _restore(cfg, checkpoint)
    model.requires_grad_(False)
    train_records, val_records = (read_manifest(p) for p in
                                 (cfg.data.train_manifest, cfg.data.val_manifest))
    for record in train_records + val_records:
        image, _ = load_pair(record, cfg.data.mask_threshold)
        center_boxes(image.width, image.height, grid)
    # Infer channels from the actual backbone; preserve baseline checkpoint compatibility.
    image, _ = load_pair(train_records[0], cfg.data.mask_threshold)
    _, features = encode(model, image, device)
    channels = features.shape[1]
    del features
    head = CorrectionHead(channels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    out.mkdir(parents=True)
    provenance = {'stage': 'shared_correction_only', 'project_commit': git_revision(),
                  'baseline_checkpoint_sha256': file_hash(checkpoint), 'data_audit': audit,
                  'epochs': epochs, 'grid': grid, 'head_grid': size, 'learning_rate': lr,
                  'preservation_weight': preservation, 'seed': cfg.training.seed,
                  'channels': channels, 'threshold': cfg.training.threshold,
                  'baseline_config': cfg.to_dict(),
                  'note': 'Frozen baseline; random centers, both actions per center; no router. '
                          'Best epoch selected by mean detail/context validation Dice.'}
    write_json(out / 'provenance.json', provenance)
    initial, _ = validate(model, head, val_records, cfg, device, grid, size)
    write_json(out / 'initial_validation.json', initial)
    # Include identity epoch as a candidate, rather than force a harmful correction.
    best = (initial['detail']['dice'] + initial['context']['dice']) / 2
    def artifact(epoch):
        return {'format_version': 1, 'kind': 'shared_correction', 'epoch': epoch,
                'head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                'provenance': provenance}
    torch.save(artifact(0), out / 'best_correction.pt')
    for epoch in range(1, epochs+1):
        head.train()
        order = list(train_records)
        random.shuffle(order)
        losses = []
        for record in order:
            image, target = load_pair(record, cfg.data.mask_threshold)
            valid = load_fov(record, target.shape)
            global_pair = encode(model, image, device)
            centers = center_boxes(image.width, image.height, grid)
            centers = [c for c in centers if valid[c[1]:c[3], c[0]:c[2]].any()]
            center = random.choice(centers)
            x0, y0, x1, y1 = center
            y = torch.tensor(target[y0:y1, x0:x1], device=device).float()[None, None]
            v = torch.tensor(valid[y0:y1, x0:x1], device=device).float()[None, None]
            native_base = F.interpolate(global_pair[0].float(), target.shape, mode='bilinear',
                                        align_corners=False)[:, :, y0:y1, x0:x1]
            optimizer.zero_grad(set_to_none=True)
            sample_loss = 0.0
            for action in (0, 1):
                inputs = region_inputs(model, image, global_pair, center, action, device, size)
                corrected = head(*inputs, action)
                native = native_base + F.interpolate(corrected-inputs[2], y.shape[-2:],
                                                     mode='bilinear', align_corners=False)
                correct = (native_base.sigmoid().ge(cfg.training.threshold) == y.bool()).float()*v
                keep = ((native.sigmoid()-native_base.sigmoid()).square()*correct).sum()
                keep = keep / correct.sum().clamp_min(1)
                loss = (segmentation_loss(native, y, v) + preservation*keep) / 2
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite correction loss')
                loss.backward()
                sample_loss += loss.item()
            nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(sample_loss)
        summary, rows = validate(model, head, val_records, cfg, device, grid, size)
        selection = (summary['detail']['dice'] + summary['context']['dice']) / 2
        if selection > best:
            best = selection
            torch.save(artifact(epoch), out / 'best_correction.pt')
        torch.save(artifact(epoch), out / 'last_correction.pt')
        record = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': summary}
        print(json.dumps(record), flush=True)
        with (out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
    saved = torch.load(out / 'best_correction.pt', map_location=device, weights_only=True)
    head.load_state_dict(saved['head'])
    summary, rows = validate(model, head, val_records, cfg, device, grid, size, out)
    write_csv(out / 'per_image.csv', rows)
    write_json(out / 'summary.json', {'best_epoch': saved['epoch'], 'modes': summary,
                                    'provenance': provenance})
    return summary
