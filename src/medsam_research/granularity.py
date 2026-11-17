"""Experimental regional supervision controller, learned using training-only meta pairs."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from .correction import encode
from .data import (
    BinaryDataset,
    audit_manifests,
    file_hash,
    load_fov,
    load_pair,
    read_manifest,
)
from .diagnostics import error_overlay, vessel_metrics
from .engine import _restore, git_revision, set_seed, write_csv, write_json
from .metrics import binary_metrics, segmentation_loss

MODES = ('pixel', 'fixed', 'curriculum', 'meta', 'reweight')


class FeatureAdapter(nn.Module):
    """Identical segmentation capacity across all supervision controls."""

    def __init__(self, channels, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature, baseline):
        delta = F.interpolate(self.net(feature), baseline.shape[-2:], mode='bilinear',
                              align_corners=False)
        return baseline + delta


class SupervisionController(nn.Module):
    def __init__(self, channels, grid=8, minimum=0.1):
        super().__init__()
        self.grid, self.minimum = grid, minimum
        self.net = nn.Sequential(nn.Conv2d(channels+2, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature, logits):
        # Features, probabilities and entropy only: no target enters the controller.
        feature = F.adaptive_avg_pool2d(feature.detach(), self.grid)
        p = logits.detach().sigmoid().clamp(1e-6, 1-1e-6)
        entropy = -(p*p.log() + (1-p)*(1-p).log())
        x = torch.cat((feature, F.adaptive_avg_pool2d(p, self.grid),
                       F.adaptive_avg_pool2d(entropy, self.grid)), 1)
        return self.minimum + (1-self.minimum)*self.net(x).sigmoid()


def regional_losses(logits, target, valid, grid):
    if grid < 1 or any(s % grid for s in logits.shape[-2:]):
        raise ValueError('Supervision grid must divide both model input dimensions')
    weights = F.adaptive_avg_pool2d(valid.float(), grid)
    active = weights > 0
    if not active.any():
        raise ValueError('No valid supervision pixels')
    denominator = weights.clamp_min(1e-8)
    pixel = F.adaptive_avg_pool2d(
        F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')*valid,
        grid) / denominator
    occupancy = F.adaptive_avg_pool2d(logits.sigmoid()*valid, grid) / denominator
    fraction = F.adaptive_avg_pool2d(target*valid, grid) / denominator
    # Both losses are BCE, avoiding an arbitrary BCE-vs-MSE scale mismatch.
    coarse = F.binary_cross_entropy(occupancy.clamp(1e-6, 1-1e-6), fraction, reduction='none')
    return pixel, coarse, active


def supervised_loss(logits, sample, alpha, grid, mode):
    pixel, coarse, active = regional_losses(logits, sample['target'], sample['valid'], grid)
    if mode == 'reweight':
        weights = alpha / alpha[active].mean().clamp_min(1e-8)
        mixed = (weights*pixel)[active].mean()
    else:
        mixed = (alpha*pixel + (1-alpha)*coarse)[active].mean()
    # Identical full-resolution soft-Dice term in every arm; only the BCE component changes.
    p, y, v = logits.sigmoid(), sample['target'], sample['valid']
    dice = (2*(p*y*v).sum()+1) / ((p*v).sum()+(y*v).sum()+1)
    return .5*mixed + .5*(1-dice)


def meta_objective(adapter, controller, support, query, grid, step_size, mode='meta'):
    logits = adapter(support['feature'], support['baseline'])
    alpha = controller(support['feature'], logits)
    inner = supervised_loss(logits, support, alpha, grid, mode)
    named = dict(adapter.named_parameters())
    gradients = torch.autograd.grad(inner, tuple(named.values()), create_graph=True)
    virtual = {key: value-step_size*g for (key, value), g in zip(named.items(), gradients)}
    query_logits = functional_call(adapter, virtual, (query['feature'], query['baseline']))
    return segmentation_loss(query_logits, query['target'], query['valid'])


def cache_samples(model, records, cfg, device):
    samples = []
    dataset = BinaryDataset(records, model.image_size, cfg.data.mask_threshold)
    for i, record in enumerate(records):
        image, _ = load_pair(record, cfg.data.mask_threshold)
        baseline, feature = encode(model, image, device)
        example = dataset[i]
        samples.append({'id': record.sample_id, 'dataset': record.dataset, 'group': record.group,
                        'feature': feature.detach().float().cpu(),
                        'baseline': baseline.detach().float().cpu(),
                        'target': example['mask'][None], 'valid': example['valid'][None]})
        print(f'Cached {record.sample_id}', flush=True)
    return samples


def to_device(sample, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()}


def query_candidates(samples):
    candidates = {}
    for i, support in enumerate(samples):
        if not support['group'] or not support['dataset']:
            raise ValueError('Meta training needs dataset and patient/video group on every row')
        eligible = [j for j, q in enumerate(samples)
                    if q['dataset'] == support['dataset'] and q['group'] != support['group']]
        if not eligible:
            raise ValueError('Each dataset needs at least two different training groups')
        candidates[i] = eligible
    return candidates


@torch.no_grad()
def validate(adapter, cached, records, cfg, device, output=None, controller=None):
    adapter.eval()
    rows, alpha_rows = [], []
    for index, (sample, record) in enumerate(zip(cached, records)):
        s = to_device(sample, device)
        logits = adapter(s['feature'], s['baseline'])
        if controller is not None:
            values = controller(s['feature'], logits)[0, 0].cpu().numpy()
            for y, x in np.ndindex(values.shape):
                alpha_rows.append({'id': record.sample_id, 'row': y, 'column': x,
                                   'pixel_weight': float(values[y, x])})
        image, target = load_pair(record, cfg.data.mask_threshold)
        valid = load_fov(record, target.shape)
        for mode, value in (('global', s['baseline']), ('adapted', logits)):
            native = F.interpolate(value, target.shape, mode='bilinear', align_corners=False)
            prediction = native[0, 0].sigmoid().ge(cfg.training.threshold).cpu().numpy()
            rows.append({'id': record.sample_id, 'dataset': record.dataset, 'group': record.group,
                         'mode': mode,
                         **binary_metrics(prediction, target, cfg.training.boundary_tolerance, valid),
                         **vessel_metrics(prediction, target, valid)})
            if output is not None:
                prefix = f'{index:06d}_{mode}'
                error_overlay(image, prediction, target, valid).save(output/f'{prefix}_errors.png')
                Image.fromarray((prediction & valid.astype(bool)).astype(np.uint8)*255).save(
                    output/f'{prefix}_mask.png')
    summary = {mode: {key: float(np.mean([r[key] for r in rows if r['mode'] == mode]))
                      for key in ('dice', 'iou', 'boundary_f1', 'precision', 'recall', 'cldice')}
               for mode in ('global', 'adapted')}
    if output is not None:
        write_csv(output/'per_image.csv', rows)
        if alpha_rows:
            write_csv(output/'validation_weights.csv', alpha_rows)
    return summary


def train_granularity(cfg, checkpoint, output, mode='meta', epochs=30, grid=8,
                      learning_rate=0.01, controller_lr=0.001, minimum=0.1):
    if mode not in MODES or epochs < 1 or grid < 1 or not 0 < minimum < 1:
        raise ValueError('Invalid mode, epochs, grid or minimum pixel weight')
    if not all(math.isfinite(v) and v > 0 for v in (learning_rate, controller_lr)):
        raise ValueError('Learning rates must be finite and positive')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    set_seed(cfg.training.seed)
    records, val_records = (read_manifest(p) for p in
                            (cfg.data.train_manifest, cfg.data.val_manifest))
    candidates = query_candidates([{'dataset': r.dataset, 'group': r.group} for r in records])
    audit = audit_manifests([cfg.data.train_manifest, cfg.data.val_manifest],
                           cfg.data.mask_threshold)
    model, device = _restore(cfg, checkpoint)
    model.requires_grad_(False)
    if model.image_size % grid:
        raise ValueError('Supervision grid must divide model input size')
    started = time.perf_counter()
    train_cache = cache_samples(model, records, cfg, device)
    val_cache = cache_samples(model, val_records, cfg, device)
    channels = train_cache[0]['feature'].shape[1]
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    adapter = FeatureAdapter(channels).to(device)
    controller = SupervisionController(channels, grid, minimum).to(device)
    # Match actual adapter SGD update to the differentiable one-step virtual update.
    optimizer = torch.optim.SGD(adapter.parameters(), lr=learning_rate)
    control_optimizer = torch.optim.Adam(controller.parameters(), lr=controller_lr)
    out.mkdir(parents=True)
    provenance = {'kind': 'regional_supervision_prototype', 'mode': mode,
                  'project_commit': git_revision(), 'baseline_config': cfg.to_dict(),
                  'baseline_checkpoint_sha256': file_hash(checkpoint), 'data_audit': audit,
                  'epochs': epochs, 'grid': grid, 'learning_rate': learning_rate,
                  'controller_lr': controller_lr, 'minimum_pixel_weight': minimum,
                  'channels': channels, 'seed': cfg.training.seed,
                  'cache_seconds': time.perf_counter()-started,
                  'note': 'Training-only same-dataset different-group meta pairs; frozen baseline; '
                          'controller controls regional BCE, full-grid soft Dice always retained; '
                          'controller is absent from segmentation inference. Novelty unconfirmed.'}
    write_json(out/'provenance.json', provenance)
    initial = validate(adapter, val_cache, val_records, cfg, device)
    write_json(out/'initial_validation.json', initial)
    best = initial['adapted']['dice']
    def artifact(epoch):
        return {'format_version': 1, 'kind': 'granularity_adapter', 'epoch': epoch,
                'adapter': {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
                'controller': {k: v.detach().cpu().clone()
                               for k, v in controller.state_dict().items()},
                'provenance': provenance}
    torch.save(artifact(0), out/'best_granularity.pt')
    history_weights = []
    training_started = time.perf_counter()
    for epoch in range(1, epochs+1):
        adapter.train()
        order = list(range(len(train_cache)))
        random.shuffle(order)
        losses, meta_losses, grad_norms = [], [], []
        for i in order:
            support = to_device(train_cache[i], device)
            # Sample for every arm, preserving identical RNG consumption and support order.
            query = to_device(train_cache[random.choice(candidates[i])], device)
            if mode in ('meta', 'reweight'):
                control_optimizer.zero_grad(set_to_none=True)
                outer = meta_objective(adapter, controller, support, query, grid,
                                       learning_rate, mode)
                gradients = torch.autograd.grad(outer, tuple(controller.parameters()))
                norm = sum(g.detach().square().sum() for g in gradients).sqrt()
                if not torch.isfinite(outer) or not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite meta objective/gradient')
                for p, gradient in zip(controller.parameters(), gradients):
                    p.grad = gradient
                control_optimizer.step()
                meta_losses.append(outer.item())
                grad_norms.append(norm.item())
            optimizer.zero_grad(set_to_none=True)
            logits = adapter(support['feature'], support['baseline'])
            with torch.no_grad():
                if mode in ('meta', 'reweight'):
                    alpha = controller(support['feature'], logits)
                else:
                    fraction = {'pixel': 1.0, 'fixed': .5,
                                'curriculum': minimum+(1-minimum)*epoch/epochs}[mode]
                    alpha = torch.full((1, 1, grid, grid), fraction, device=device)
            loss = supervised_loss(logits, support, alpha, grid, mode)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adaptation loss')
            loss.backward()
            # No clipping/momentum: actual and virtual SGD have the same update rule.
            if any(not torch.isfinite(p.grad).all() for p in adapter.parameters()):
                raise FloatingPointError('Nonfinite adapter gradient')
            optimizer.step()
            losses.append(loss.item())
            a = alpha[0, 0].cpu().numpy()
            for y, x in np.ndindex(a.shape):
                history_weights.append({'epoch': epoch, 'id': support['id'], 'row': y,
                                        'column': x, 'pixel_weight': float(a[y, x])})
        summary = validate(adapter, val_cache, val_records, cfg, device)
        record = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': summary,
                  'meta_loss': float(np.mean(meta_losses)) if meta_losses else None,
                  'controller_gradient_norm': float(np.mean(grad_norms)) if grad_norms else None,
                  'pixel_weight_mean': float(np.mean([r['pixel_weight'] for r in history_weights
                                                      if r['epoch'] == epoch]))}
        print(json.dumps(record), flush=True)
        with (out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        if summary['adapted']['dice'] > best:
            best = summary['adapted']['dice']
            torch.save(artifact(epoch), out/'best_granularity.pt')
        torch.save(artifact(epoch), out/'last_granularity.pt')
    saved = torch.load(out/'best_granularity.pt', map_location=device, weights_only=True)
    adapter.load_state_dict(saved['adapter'])
    controller.load_state_dict(saved['controller'])
    summary = validate(adapter, val_cache, val_records, cfg, device, out,
                       controller if mode in ('meta', 'reweight') else None)
    write_csv(out/'training_weights.csv', history_weights)
    timing = {'training_and_validation_seconds': time.perf_counter()-training_started}
    if device.type == 'cuda':
        timing['peak_allocated_bytes_after_cache'] = torch.cuda.max_memory_allocated(device)
    write_json(out/'summary.json', {'best_epoch': saved['epoch'], 'modes': summary,
                                   'timing': timing, 'provenance': provenance})
    return summary


def evaluate_granularity(cfg, checkpoint, adapter_checkpoint, manifest, output):
    """Evaluate a fixed artifact; test labels never drive the controller or updates."""
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    saved = torch.load(adapter_checkpoint, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'granularity_adapter' or saved.get('format_version') != 1:
        raise ValueError('Unsupported granularity adapter')
    if saved['provenance']['baseline_checkpoint_sha256'] != file_hash(checkpoint):
        raise ValueError('Adapter was trained against a different baseline checkpoint')
    previous = saved['provenance']['baseline_config']
    if (previous['model']['sam_config'] != cfg.model.sam_config
            or previous['model']['prompt_tokens'] != cfg.model.prompt_tokens
            or previous['data']['mask_threshold'] != cfg.data.mask_threshold
            or previous['training']['threshold'] != cfg.training.threshold
            or previous['training']['boundary_tolerance'] != cfg.training.boundary_tolerance):
        raise ValueError('Adapter evaluation settings differ from training')
    paths = [cfg.data.train_manifest, cfg.data.val_manifest]
    if Path(manifest).resolve() != Path(cfg.data.val_manifest).resolve():
        paths.append(manifest)
    audit = audit_manifests(paths, cfg.data.mask_threshold)
    model, device = _restore(cfg, checkpoint)
    model.requires_grad_(False)
    records = read_manifest(manifest)
    cached = cache_samples(model, records, cfg, device)
    adapter = FeatureAdapter(saved['provenance']['channels']).to(device)
    adapter.load_state_dict(saved['adapter'], strict=True)
    out.mkdir(parents=True)
    summary = validate(adapter, cached, records, cfg, device, out)
    write_json(out/'summary.json', {'modes': summary, 'data_audit': audit,
                                   'adapter_sha256': file_hash(adapter_checkpoint),
                                   'baseline_checkpoint_sha256': file_hash(checkpoint),
                                   'best_epoch': saved['epoch'],
                                   'note': 'Fixed adapter, no controller or test-time updates.'})
    return summary
