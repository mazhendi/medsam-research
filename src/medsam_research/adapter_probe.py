"""Training-only small-set learnability probe; never evaluates/selects on test labels."""

import math
import random
from pathlib import Path

import torch

from .data import audit_manifests, file_hash, read_manifest
from .engine import _restore, git_revision, set_seed, write_csv, write_json
from .granularity import FeatureAdapter, cache_samples, supervised_loss
from .metrics import segmentation_loss


@torch.no_grad()
def measurements(adapter, sample, threshold):
    logits = adapter(sample['feature'], sample['baseline'])
    baseline = sample['baseline']
    y, v = sample['target'].bool(), sample['valid'].bool()
    p = logits.sigmoid().ge(threshold) & v
    y = y & v
    denom = (p.sum()+y.sum()).item()
    delta = (logits-baseline)[v]
    return {'loss': segmentation_loss(logits, sample['target'], sample['valid']).item(),
            'dice_model_grid': 2*(p & y).sum().item()/denom if denom else 1.,
            'residual_abs_mean': delta.abs().mean().item(),
            'residual_abs_max': delta.abs().max().item()}


def fit_probe(sample, channels, steps, optimizer_name, lr, seed, threshold):
    """Full small-set batch and identical initial weights for every optimizer arm."""
    set_seed(seed)
    adapter = FeatureAdapter(channels).to(sample['feature'].device)
    optimizer = (torch.optim.SGD(adapter.parameters(), lr=lr) if optimizer_name == 'sgd'
                 else torch.optim.Adam(adapter.parameters(), lr=lr))
    initial = {name: p.detach().clone() for name, p in adapter.named_parameters()}
    rows = [{'step': 0, **measurements(adapter, sample, threshold),
             'gradient_norm': None, 'parameter_update_norm': 0., 'parameter_drift_norm': 0.,
             'update_to_parameter_ratio': 0.}]
    layer_rows = []
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        logits = adapter(sample['feature'], sample['baseline'])
        loss = segmentation_loss(logits, sample['target'], sample['valid'])
        if not torch.isfinite(loss):
            return rows, layer_rows, f'nonfinite loss before step {step}'
        loss.backward()
        params = dict(adapter.named_parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):
            return rows, layer_rows, f'missing/nonfinite gradient at step {step}'
        gradient_norm = sum(p.grad.detach().square().sum() for p in params.values()).sqrt().item()
        before = {name: p.detach().clone() for name, p in params.items()}
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params.values()):
            return rows, layer_rows, f'nonfinite parameter after step {step}'
        update = sum((p.detach()-before[name]).square().sum()
                     for name, p in params.items()).sqrt().item()
        magnitude = sum(v.square().sum() for v in before.values()).sqrt().item()
        drift = sum((p.detach()-initial[name]).square().sum()
                    for name, p in params.items()).sqrt().item()
        if step == 1 or step == steps or step % 10 == 0:
            scores = measurements(adapter, sample, threshold)
            if not all(math.isfinite(value) for value in scores.values()):
                return rows, layer_rows, f'nonfinite measurement after step {step}'
            rows.append({'step': step, **scores, 'gradient_norm': gradient_norm,
                         'parameter_update_norm': update, 'parameter_drift_norm': drift,
                         'update_to_parameter_ratio': update/max(magnitude, 1e-12)})
            for name, p in params.items():
                layer_rows.append({'step': step, 'parameter': name,
                                   'gradient_norm': p.grad.norm().item(),
                                   'update_norm': (p.detach()-before[name]).norm().item(),
                                   'drift_norm': (p.detach()-initial[name]).norm().item()})
    return rows, layer_rows, None



def supervision_gradient_probe(sample, channels, seed, grid=8):
    """Test whether the two supervision choices produce distinguishable updates."""
    set_seed(seed)
    adapter = FeatureAdapter(channels).to(sample['feature'].device)
    vectors, losses = [], []
    for fraction in (0., 1.):
        logits = adapter(sample['feature'], sample['baseline'])
        alpha = torch.full((len(logits), 1, grid, grid), fraction, device=logits.device)
        loss = supervised_loss(logits, sample, alpha, grid, 'fixed')
        gradients = torch.autograd.grad(loss, tuple(adapter.parameters()))
        vectors.append(torch.cat([g.detach().reshape(-1) for g in gradients]))
        losses.append(loss.item())
    coarse, pixel = vectors
    norm_c, norm_p = coarse.norm().item(), pixel.norm().item()
    denominator = norm_c*norm_p
    return {'coarse_plus_dice_loss': losses[0], 'pixel_plus_dice_loss': losses[1],
            'coarse_gradient_norm': norm_c, 'pixel_gradient_norm': norm_p,
            'gradient_cosine': (coarse@pixel).item()/denominator if denominator > 0 else None,
            'gradient_difference_norm': (coarse-pixel).norm().item(),
            'note': 'Initial head, full probe batch; both choices retain fixed Dice. '
                    'Different loss values alone do not prove different useful updates.'}


def probe_adapter(cfg, checkpoint, output, steps=100, samples=2,
                  sgd_rates=(0.01, 0.1, 1.0), adam_lr=0.001):
    if steps < 1 or samples < 1 or not sgd_rates:
        raise ValueError('Positive steps/samples and at least one SGD rate required')
    if any(not math.isfinite(v) or v <= 0 for v in (*sgd_rates, adam_lr)):
        raise ValueError('Learning rates must be positive and finite')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    audit = audit_manifests([cfg.data.train_manifest, cfg.data.val_manifest],
                           cfg.data.mask_threshold)
    records = read_manifest(cfg.data.train_manifest)
    rng = random.Random(cfg.training.seed)
    order = list(records)
    rng.shuffle(order)
    # Prefer distinct groups. This is a fitting diagnostic, not a generalization experiment.
    chosen, seen = [], set()
    for record in order:
        key = (record.dataset, record.group) if record.group else ('id', record.sample_id)
        if key not in seen:
            chosen.append(record)
            seen.add(key)
        if len(chosen) == samples:
            break
    if len(chosen) < samples:
        raise ValueError('Not enough distinct training groups/images for requested probe samples')
    model, device = _restore(cfg, checkpoint)
    model.requires_grad_(False)
    cached = cache_samples(model, chosen, cfg, device)
    del model
    batch = {key: torch.cat([s[key] for s in cached]).to(device)
             for key in ('feature', 'baseline', 'target', 'valid')}
    # Record raw feature scale: no untested normalization is silently introduced.
    feature_stats = {key: float(value) for key, value in {
        'mean': batch['feature'].mean(), 'std': batch['feature'].std(),
        'abs_max': batch['feature'].abs().max(), 'rms': batch['feature'].square().mean().sqrt(),
        'baseline_abs_mean': batch['baseline'].abs().mean(),
    }.items()}
    contrast = supervision_gradient_probe(batch, batch['feature'].shape[1], cfg.training.seed)
    rows, layers, summaries = [], [], []
    arms = [('sgd', lr) for lr in sgd_rates] + [('adam', adam_lr)]
    for index, (optimizer, lr) in enumerate(arms):
        arm = f'{index}_{optimizer}_{lr:g}'
        result, per_layer, error = fit_probe(batch, batch['feature'].shape[1], steps,
                                           optimizer, lr, cfg.training.seed, cfg.training.threshold)
        rows.extend({'arm': arm, 'optimizer': optimizer, 'learning_rate': lr, **r} for r in result)
        layers.extend({'arm': arm, **r} for r in per_layer)
        initial, final = result[0], result[-1]
        summaries.append({'arm': arm, 'error': error, 'completed_steps': final['step'],
                          'initial': initial, 'final': final,
                          'relative_loss_reduction': (initial['loss']-final['loss']) /
                                                     max(initial['loss'], 1e-12)})
        print(f"{arm}: {initial['loss']:.6f} -> {final['loss']:.6f}; error={error}", flush=True)
    out.mkdir(parents=True)
    write_csv(out/'probe_history.csv', rows)
    if layers:
        write_csv(out/'per_layer.csv', layers)
    write_json(out/'summary.json', {
        'kind': 'training_only_adapter_probe', 'project_commit': git_revision(),
        'baseline_checkpoint_sha256': file_hash(checkpoint), 'data_audit': audit,
        'sample_ids': [r.sample_id for r in chosen], 'steps': steps,
        'feature_stats': feature_stats, 'supervision_gradient_contrast': contrast, 'arms': summaries,
        'note': 'Same small full batch and identical initialization in all arms. '
                'Dice is pooled over selected training images at model input resolution. '
                'No validation scoring or controller. This probes fitting, not generalization; '
                'optimizer choices here are not yet implemented in meta training.'})
    return summaries
