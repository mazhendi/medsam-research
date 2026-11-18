"""Pure supervised controls: continue baseline versus deep/multi-scale residual heads."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .data import audit_manifests, file_hash, load_fov, load_pair, read_manifest
from .diagnostics import error_overlay, vessel_metrics
from .engine import _restore, git_revision, make_loader, set_seed, write_csv, write_json
from .granularity import FeatureAdapter
from .metrics import binary_metrics, native_prediction, segmentation_loss

MODES = ('decoder', 'deep', 'pooled', 'multiscale')


class MultiScaleHead(nn.Module):
    def __init__(self, channels, width=16):
        super().__init__()
        if len(channels) != 3:
            raise ValueError('Expected three SAM2 feature levels, fine to coarse')
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.fusions = nn.ModuleList(nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                                  nn.GELU()) for _ in range(2))
        self.output = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pyramid, baseline):
        x = self.projections[2](pyramid[2])
        for fusion, level in zip(self.fusions, (1, 0)):
            x = F.interpolate(x, pyramid[level].shape[-2:], mode='bilinear', align_corners=False)
            x = fusion(x+self.projections[level](pyramid[level]))
        delta = F.interpolate(self.output(x), baseline.shape[-2:], mode='bilinear',
                              align_corners=False)
        return baseline+delta


class BackboneControl(nn.Module):
    def __init__(self, baseline, mode, channels, width=16):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown backbone control mode')
        self.baseline, self.mode = baseline, mode
        self.image_size = baseline.image_size
        if mode == 'decoder':
            # Restore retains exactly the baseline's original trainable decoder/tokens.
            self.head = None
        else:
            self.baseline.requires_grad_(False)
            self.head = (FeatureAdapter(channels[-1]) if mode == 'deep'
                         else MultiScaleHead(channels, width))

    def train(self, mode=True):
        super().train(mode)
        if self.mode != 'decoder':
            self.baseline.eval()
        return self

    def forward(self, image):
        if self.mode == 'decoder':
            return self.baseline(image)
        with torch.no_grad():
            logits, pyramid = self.baseline.forward_with_pyramid(image)
        if self.mode == 'deep':
            return self.head(pyramid[-1], logits)
        if self.mode == 'pooled':
            # Same head/parameters as multiscale; remove fine spatial variation only
            # from the new head's inputs. Original SAM decoder remains unchanged.
            coarse = pyramid[-1].shape[-2:]
            pyramid = [F.interpolate(F.adaptive_avg_pool2d(p, coarse), p.shape[-2:],
                                     mode='bilinear', align_corners=False) for p in pyramid]
        return self.head(pyramid, logits)


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if p.requires_grad}


@torch.no_grad()
def score_model(model, loader, device, cfg, collect=False):
    model.eval()
    rows, masks = [], {}
    for batch in loader:
        logits = model(batch['image'].to(device))
        for i, sid in enumerate(batch['id']):
            y, valid = batch['native_mask'][i], batch['native_valid'][i]
            p = native_prediction(logits[i:i+1], y.shape, cfg.training.threshold)
            rows.append({'id': sid, 'fov_used': batch['fov_used'][i],
                         **binary_metrics(p, y, cfg.training.boundary_tolerance, valid),
                         **vessel_metrics(p, y, valid)})
            if collect:
                masks[sid] = p
    keys = ('dice', 'iou', 'boundary_f1', 'precision', 'recall', 'cldice',
            'thin_skeleton_recall', 'medium_skeleton_recall', 'thick_skeleton_recall')
    summary = {}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary['samples'] = len(rows)
    return summary, rows, masks


def export_comparison(out, records, cfg, baseline_rows, baseline_masks, rows, masks):
    combined, deltas = [], []
    for index, (record, before, after) in enumerate(zip(records, baseline_rows, rows)):
        if record.sample_id != before['id'] or before['id'] != after['id']:
            raise RuntimeError('Validation sample order changed')
        image, target = load_pair(record, cfg.data.mask_threshold)
        valid = load_fov(record, target.shape)
        for mode, values, p in (('baseline', before, baseline_masks[record.sample_id]),
                                ('adapted', after, masks[record.sample_id])):
            combined.append({'id': record.sample_id, 'dataset': record.dataset,
                             'group': record.group, 'mode': mode,
                             **{k: v for k, v in values.items() if k != 'id'}})
            prefix = f'{index:06d}_{mode}'
            error_overlay(image, p, target, valid).save(out/f'{prefix}_errors.png')
            Image.fromarray((p & valid.astype(bool)).astype(np.uint8)*255).save(
                out/f'{prefix}_mask.png')
        deltas.append({'id': record.sample_id, 'dataset': record.dataset, 'group': record.group,
                       **{f'delta_{k}': after[k]-before[k] for k in
                          ('dice', 'precision', 'recall', 'boundary_f1', 'cldice')}})
    write_csv(out/'per_image.csv', combined)
    write_csv(out/'paired_deltas.csv', deltas)


def train_backbone_control(cfg, checkpoint, output, mode='multiscale', epochs=50,
                           learning_rate=None, width=16):
    if mode not in MODES or epochs < 1 or width < 1:
        raise ValueError('Invalid control mode, epochs or width')
    if learning_rate is None:
        learning_rate = 1e-4 if mode == 'decoder' else 1e-3
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Learning rate must be positive and finite')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    set_seed(cfg.training.seed)
    audit = audit_manifests([cfg.data.train_manifest, cfg.data.val_manifest],
                           cfg.data.mask_threshold)
    baseline, device = _restore(cfg, checkpoint)
    train_records, val_records = (read_manifest(p) for p in
                                 (cfg.data.train_manifest, cfg.data.val_manifest))
    train_loader = make_loader(train_records, baseline.image_size, cfg, True)
    fit_loader = make_loader(train_records, baseline.image_size, cfg)
    val_loader = make_loader(val_records, baseline.image_size, cfg)
    with torch.no_grad():
        image = next(iter(fit_loader))['image'][:1].to(device)
        _, pyramid = baseline.forward_with_pyramid(image)
        channels = [p.shape[1] for p in pyramid]
        shapes = [list(p.shape[1:]) for p in pyramid]
    del image, pyramid
    model = BackboneControl(baseline, mode, channels, width).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=cfg.training.weight_decay)
    use_amp = bool(cfg.training.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    out.mkdir(parents=True)
    provenance = {'kind': 'backbone_control', 'mode': mode, 'project_commit': git_revision(),
                  'baseline_checkpoint_sha256': file_hash(checkpoint), 'baseline_config': cfg.to_dict(),
                  'data_audit': audit, 'epochs': epochs, 'learning_rate': learning_rate,
                  'optimizer': 'AdamW (fresh state)', 'width': width, 'channels': channels,
                  'pyramid_shapes': shapes, 'trainable_parameters': sum(p.numel() for p in parameters),
                  'seed': cfg.training.seed, 'amp_training': use_amp,
                  'note': 'Pure BCE+Dice, no controller/crops; all modes run baseline online. '
                          'Equal epochs/updates are not equal parameters or computation. '
                          'Decoder defaults lr=1e-4, residual heads lr=1e-3.'}
    write_json(out/'provenance.json', provenance)
    initial, baseline_rows, baseline_masks = score_model(model, val_loader, device, cfg, True)
    write_json(out/'initial_validation.json', initial)
    def artifact(epoch):
        return {'format_version': 1, 'kind': 'backbone_control', 'epoch': epoch,
                'learned': trainable_state(model), 'provenance': provenance}
    torch.save(artifact(0), out/'best_control.pt')
    best = initial['dice']
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs+1):
        model.train()
        loss_total, seen, grad_norms = 0., 0, []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch['image'].to(device))
                loss = segmentation_loss(logits, batch['mask'].to(device), batch['valid'].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite control loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            grad_norms.append(norm.item())
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item()*len(logits)
            seen += len(logits)
        # Training/validation metrics use native labels and FP32 inference in every arm.
        training, _, _ = score_model(model, fit_loader, device, cfg)
        validation, rows, _ = score_model(model, val_loader, device, cfg)
        record = {'epoch': epoch, 'train_loss': loss_total/seen,
                  'training': training, 'validation': validation,
                  'gradient_norm_before_clip': float(np.mean(grad_norms))}
        print(json.dumps(record), flush=True)
        with (out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        if validation['dice'] > best:
            best = validation['dice']
            torch.save(artifact(epoch), out/'best_control.pt')
        torch.save(artifact(epoch), out/'last_control.pt')
    saved = torch.load(out/'best_control.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['learned'], strict=False)
    final, rows, masks = score_model(model, val_loader, device, cfg, True)
    export_comparison(out, val_records, cfg, baseline_rows, baseline_masks, rows, masks)
    timing = {'training_and_scoring_seconds': time.perf_counter()-started}
    if device.type == 'cuda':
        timing['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    write_json(out/'summary.json', {'best_epoch': saved['epoch'],
                                   'modes': {'baseline': initial, 'adapted': final},
                                   'timing': timing, 'provenance': provenance})
    return final


def evaluate_backbone_control(cfg, checkpoint, control_checkpoint, manifest, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    saved = torch.load(control_checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1 or saved.get('kind') != 'backbone_control':
        raise ValueError('Unsupported backbone control checkpoint')
    provenance = saved['provenance']
    if provenance['baseline_checkpoint_sha256'] != file_hash(checkpoint):
        raise ValueError('Control was trained against a different baseline')
    previous = provenance['baseline_config']
    if (previous['model']['sam_config'] != cfg.model.sam_config
            or previous['model']['prompt_tokens'] != cfg.model.prompt_tokens
            or previous['data']['mask_threshold'] != cfg.data.mask_threshold
            or previous['training']['threshold'] != cfg.training.threshold
            or previous['training']['boundary_tolerance'] != cfg.training.boundary_tolerance):
        raise ValueError('Control evaluation settings differ from training')
    paths = [cfg.data.train_manifest, cfg.data.val_manifest]
    if Path(manifest).resolve() != Path(cfg.data.val_manifest).resolve():
        paths.append(manifest)
    audit = audit_manifests(paths, cfg.data.mask_threshold)
    baseline, device = _restore(cfg, checkpoint)
    model = BackboneControl(baseline, provenance['mode'], provenance['channels'],
                            provenance['width']).to(device)
    if set(saved['learned']) != {name for name, p in model.named_parameters() if p.requires_grad}:
        raise ValueError('Missing or unexpected learned control parameters')
    records = read_manifest(manifest)
    loader = make_loader(records, model.image_size, cfg)
    initial, baseline_rows, baseline_masks = score_model(model, loader, device, cfg, True)
    model.load_state_dict(saved['learned'], strict=False)
    final, rows, masks = score_model(model, loader, device, cfg, True)
    out.mkdir(parents=True)
    export_comparison(out, records, cfg, baseline_rows, baseline_masks, rows, masks)
    write_json(out/'summary.json', {'modes': {'baseline': initial, 'adapted': final},
                                   'best_epoch': saved['epoch'], 'data_audit': audit,
                                   'control_sha256': file_hash(control_checkpoint),
                                   'baseline_checkpoint_sha256': file_hash(checkpoint),
                                   'note': 'Fixed weights; no test-time updates or controller.'})
    return final
