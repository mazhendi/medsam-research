"""Isolated train/evaluate/predict/probe workflow for the V3 mechanism prototype."""
import json
import time
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import engine
from .config import SAM2_COMMIT
from .data import audit_manifests, file_hash, image_tensor, load_pair, read_manifest
from .metrics import binary_metrics, native_prediction
from .region_shape import RegionShapeConfig, RegionShapeSAM2, region_shape_loss

KIND = 'region_shape_sam2_v1'


def read_artifact(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('kind') != KIND or saved.get('format_version') != 1:
        raise ValueError('Expected a region-shape checkpoint, not a baseline or probe')
    if saved.get('sam2_commit') != SAM2_COMMIT:
        raise ValueError('SAM2 source version mismatch')
    return saved


def mechanism_state_names(model):
    # Frozen RS parameters must be saved too when only B is optimized.
    return {name for name, p in model.named_parameters()
            if (not name.startswith('baseline.') or p.requires_grad
                or (model.config.multiscale_refinement
                    and name.startswith('baseline.sam.sam_mask_decoder.')))}


def mechanism_state(model):
    names = mechanism_state_names(model)
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if name in names}


def build_system(cfg, baseline_checkpoint, mechanism=None, checkpoint=None):
    if not Path(baseline_checkpoint).is_file():
        raise FileNotFoundError(baseline_checkpoint)
    saved = read_artifact(checkpoint) if checkpoint else None
    if saved:
        if saved['baseline_sha256'] != file_hash(baseline_checkpoint):
            raise ValueError('Different baseline checkpoint from the training run')
        if saved['evaluation'] != {'threshold': cfg.training.threshold,
                                   'mask_threshold': cfg.data.mask_threshold,
                                   'boundary_tolerance': cfg.training.boundary_tolerance}:
            raise ValueError('Evaluation settings differ from training')
        previous = RegionShapeConfig(**saved['mechanism']).validate().to_dict()
        if mechanism is not None and mechanism.to_dict() != previous:
            raise ValueError('Mechanism settings differ from checkpoint')
        mechanism = RegionShapeConfig(**previous).validate()
    if mechanism is None:
        raise ValueError('Mechanism configuration is required')
    baseline, device = engine._restore(cfg, baseline_checkpoint)
    model = RegionShapeSAM2(baseline, mechanism).to(device)
    if saved:
        expected = mechanism_state_names(model)
        if set(saved['adapter']) != expected:
            raise ValueError('Unexpected or missing mechanism/decoder parameters')
        # Shape errors raise in load_state_dict; missing frozen base parameters
        # are intentional, supplied by the bound baseline checkpoint.
        model.load_state_dict(saved['adapter'], strict=False)
    return model, device, saved


def initialize_feature_extension(model, cfg, baseline_checkpoint, checkpoint):
    """Load the complete old mechanism; only new feature-module keys may be absent."""
    previous, _, _ = build_system(cfg, baseline_checkpoint, checkpoint=checkpoint)
    old = previous.config.to_dict()
    new = model.config.to_dict()
    switches = {'scale_selection', 'detail_retrieval', 'detail_window', 'detail_bias',
                'multiscale_refinement', 'refinement_width', 'refinement_window',
                'refinement_only', 'train_decoder'}
    if any(old[k] != new[k] for k in old.keys()-switches):
        raise ValueError('Warm start must preserve the original region-shape settings')
    if old['scale_selection'] or old['detail_retrieval'] or old['multiscale_refinement']:
        raise ValueError('Warm start expects region-shape without feature modules')
    model.load_state_dict(previous.state_dict(), strict=False)


def artifact(model, cfg, baseline_checkpoint, epoch, scores, source_audit):
    return {'kind': KIND, 'format_version': 1, 'sam2_commit': SAM2_COMMIT,
            'baseline_sha256': file_hash(baseline_checkpoint),
            'mechanism': model.config.to_dict(), 'adapter': mechanism_state(model),
            'epoch': epoch, 'validation': scores, 'source_config': cfg.to_dict(), 'source_audit': source_audit,
            'evaluation': {'threshold': cfg.training.threshold,
                           'mask_threshold': cfg.data.mask_threshold,
                           'boundary_tolerance': cfg.training.boundary_tolerance}}


def precision_recall(prediction, target, valid):
    p, y, v = prediction.astype(bool), target.astype(bool), valid.astype(bool)
    tp = int((p & y & v).sum())
    ps, ys = int((p & v).sum()), int((y & v).sum())
    return {'precision': tp/ps if ps else float(ys == 0),
            'recall': tp/ys if ys else float(ps == 0)}


@torch.no_grad()
def score(model, loader, cfg, device, output=None):
    model.eval()
    rows, diagnostic_rows, changes = [], [], []
    counter = 0
    for batch in loader:
        result = model.forward_details(batch['image'].to(device))
        for i, sid in enumerate(batch['id']):
            y, v = batch['native_mask'][i], batch['native_valid'][i]
            predictions = {}
            for mode, key in [('baseline', 'baseline_logits'), ('region_shape', 'logits')]:
                prediction = native_prediction(result[key][i:i+1], y.shape, cfg.training.threshold)
                predictions[mode] = prediction
                rows.append({'id': sid, 'mode': mode,
                             **binary_metrics(prediction, y, cfg.training.boundary_tolerance, v),
                             **precision_recall(prediction, y, v)})
                if output:
                    Image.fromarray(prediction.astype(np.uint8)*255).save(
                        Path(output)/f'{counter:06d}_{mode}_mask.png')
            old, new = predictions['baseline'], predictions['region_shape']
            truth, region = y.astype(bool), v.astype(bool)
            corrected = (old != truth) & (new == truth) & region
            damaged = (old == truth) & (new != truth) & region
            changes.append({'id': sid, 'corrected_pixels': int(corrected.sum()),
                            'damaged_pixels': int(damaged.sum()),
                            'corrected_foreground': int((corrected & truth).sum()),
                            'corrected_background': int((corrected & ~truth).sum()),
                            'lost_foreground': int((damaged & truth).sum()),
                            'new_false_positive': int((damaged & ~truth).sum())})
            weights = result['weights'][i].float().cpu()
            diagnostic_rows.append({'id': sid, 'weights': weights.tolist(),
                'weight_entropy': float(-(weights*weights.clamp_min(1e-8).log()).sum()),
                'mean_disagreement': float(result['variance'][i].mean()),
                'mean_spatial_weights': result['spatial_weights'][i].mean((-2, -1)).cpu().tolist(),
                'spatial_weight_std': result['spatial_weights'][i].flatten(1).std(-1, unbiased=False).cpu().tolist(),
                'hypothesis_foreground': result['occupancy'][i].mean((-2, -1)).cpu().tolist()})
            if output:
                # Floating-point fields are exported separately from visual PNGs.
                np.savez_compressed(Path(output)/f'{counter:06d}_states.npz',
                    fields=result['field'][i].float().cpu().numpy(),
                    spatial_weights=result['spatial_weights'][i].float().cpu().numpy(),
                    occupancy=result['occupancy'][i].float().cpu().numpy(),
                    region_probability=result['probability'][i].float().cpu().numpy(),
                    consensus=result['mean'][i].float().cpu().numpy(),
                    disagreement=result['variance'][i].float().cpu().numpy())
            counter += 1
    if not rows:
        raise ValueError('Empty evaluation split')
    summary = {}
    for mode in ('baseline', 'region_shape'):
        selected = [r for r in rows if r['mode'] == mode]
        summary[mode] = {key: float(np.mean([r[key] for r in selected])) for key in
            ('dice', 'iou', 'boundary_f1', 'precision', 'recall', 'predicted_foreground_fraction')}
        summary[mode]['samples'] = len(selected)
    if output:
        engine.write_csv(Path(output)/'per_image.csv', rows)
        engine.write_csv(Path(output)/'paired_changes.csv', changes)
        engine.write_json(Path(output)/'states.json', diagnostic_rows)
    return summary, rows


def amp_settings(cfg, device):
    enabled = bool(cfg.training.amp and device.type == 'cuda')
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled, dtype, torch.amp.GradScaler('cuda', enabled=enabled and dtype == torch.float16)


def update(model, batch, device, optimizer, scaler, enabled, dtype):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        result = model.forward_details(batch['image'].to(device))
        losses = region_shape_loss(result, batch['mask'].to(device),
                                   batch['valid'].to(device), model.config)
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Nonfinite co-inference loss')
    scaler.scale(losses['total']).backward()
    scaler.unscale_(optimizer)
    gradients = {}
    for name in ('head', 'dense_prompt', 'sparse_prompt', 'task_query',
                 'scale_module', 'detail_module', 'refinement'):
        if not hasattr(model, name):
            continue
        grad = [p.grad.detach().float().square().sum() for p in getattr(model, name).parameters()
                if p.grad is not None]
        gradients[name] = float(torch.stack(grad).sum().sqrt()) if grad else 0.
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  1., error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return {name: value.detach().item() for name, value in losses.items()}, gradients


def train_region_shape(cfg, baseline_checkpoint, mechanism, output, epochs=None,
                       initialize_from=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    epochs = cfg.training.epochs if epochs is None else epochs
    if epochs < 1:
        raise ValueError('epochs must be positive')
    engine.set_seed(cfg.training.seed)
    audit = audit_manifests([cfg.data.train_manifest, cfg.data.val_manifest], cfg.data.mask_threshold)
    model, device, _ = build_system(cfg, baseline_checkpoint, mechanism)
    if mechanism.refinement_only and not initialize_from:
        raise ValueError('B-only training requires --initialize-from with trained RS weights')
    if initialize_from:
        initialize_feature_extension(model, cfg, baseline_checkpoint, initialize_from)
    training = engine.make_loader(read_manifest(cfg.data.train_manifest), model.image_size, cfg,
                                  True, augment=cfg.training.augmentation != 'none')
    validation = engine.make_loader(read_manifest(cfg.data.val_manifest), model.image_size, cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    enabled, dtype, scaler = amp_settings(cfg, device)
    out.mkdir(parents=True)
    versions = {}
    for package in ('torch', 'SAM-2', 'numpy', 'scipy', 'Pillow'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    engine.write_json(out/'provenance.json', {'project_commit': engine.git_revision(),
        'project_dirty': engine.git_dirty(), 'source_config': cfg.to_dict(),
        'mechanism': mechanism.to_dict(), 'baseline_sha256': file_hash(baseline_checkpoint),
        'pretrained_sha256': file_hash(cfg.model.checkpoint), 'data_audit': audit,
        'initialize_from_sha256': file_hash(initialize_from) if initialize_from else None,
        'versions': versions, 'epochs': epochs, 'training_precision': str(dtype) if enabled else 'float32',
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'encoder_passes_per_image': 1, 'decoder_passes_per_image': 2,
        'note': 'Prototype. Initial baseline equality is prompt initialization, not logit residual addition.'})
    initial, _ = score(model, validation, cfg, device)
    engine.write_json(out/'initial_validation.json', initial)
    best, best_epoch = initial['region_shape']['dice'], 0
    torch.save(artifact(model, cfg, baseline_checkpoint, 0, initial, audit), out/'best_region_shape.pt')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        totals, norms, samples = {}, {}, 0
        for batch in training:
            loss, gradients = update(model, batch, device, optimizer, scaler, enabled, dtype)
            count = len(batch['id'])
            for name, value in loss.items():
                totals[name] = totals.get(name, 0.)+value*count
            for name, value in gradients.items():
                norms[name] = norms.get(name, 0.)+value*count
            samples += count
        scores, _ = score(model, validation, cfg, device)
        record = {'epoch': epoch, 'losses': {k: v/samples for k, v in totals.items()},
                  'gradient_norms': {k: v/samples for k, v in norms.items()}, 'validation': scores}
        print(json.dumps(record, allow_nan=False), flush=True)
        with (out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        saved = artifact(model, cfg, baseline_checkpoint, epoch, scores, audit)
        torch.save(saved, out/'last_region_shape.pt')
        if scores['region_shape']['dice'] > best:
            best, best_epoch = scores['region_shape']['dice'], epoch
            torch.save(saved, out/'best_region_shape.pt')
    best_saved = read_artifact(out/'best_region_shape.pt')
    model.load_state_dict(best_saved['adapter'], strict=False)
    scores, _ = score(model, validation, cfg, device, out)
    engine.write_json(out/'training_summary.json', {'best_epoch': best_epoch, 'validation': scores,
        'wall_seconds': time.perf_counter()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'baseline_selected': best_epoch == 0})
    if best_epoch == 0:
        print('Validation selected epoch 0: no improvement over identity initialization established.', flush=True)
    return out


def evaluate_region_shape(cfg, baseline_checkpoint, checkpoint, manifest, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    saved = read_artifact(checkpoint)
    # Audit against manifests saved at training time, not replacement paths in a
    # current config, so changing configs cannot bypass held-out data checks.
    source = saved['source_config']['data']
    paths = list(dict.fromkeys(str(Path(p).resolve()) for p in
                               (source['train_manifest'], source['val_manifest'], manifest)))
    audit = audit_manifests(paths, cfg.data.mask_threshold)
    current = {r['manifest']: r for r in audit}
    for expected in saved['source_audit']:
        actual = current[expected['manifest']]
        if any(actual[key] != expected[key] for key in ('sha256', 'decoded_data_sha256')):
            raise ValueError('Source training/validation data changed since training')
    model, device, _ = build_system(cfg, baseline_checkpoint, checkpoint=checkpoint)
    loader = engine.make_loader(read_manifest(manifest), model.image_size, cfg)
    out.mkdir(parents=True)
    scores, _ = score(model, loader, cfg, device, out)
    engine.write_json(out/'summary.json', {'modes': scores, 'audit': audit,
        'checkpoint_sha256': file_hash(checkpoint), 'baseline_sha256': file_hash(baseline_checkpoint)})
    return scores


@torch.no_grad()
def predict_region_shape(cfg, baseline_checkpoint, checkpoint, manifest, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    model, device, _ = build_system(cfg, baseline_checkpoint, checkpoint=checkpoint)
    model.eval()
    records = read_manifest(manifest, require_masks=False)
    out.mkdir(parents=True)
    rows = []
    for i, record in enumerate(records):
        image, _ = load_pair(replace(record, mask=None, fov=None))
        tensor = image_tensor(image, model.image_size)[None].to(device)
        result = model.forward_details(tensor)
        mask = native_prediction(result['logits'], (image.height, image.width), cfg.training.threshold)
        filename = f'{i:06d}.png'
        Image.fromarray(mask.astype(np.uint8)*255).save(out/filename)
        rows.append({'id': record.sample_id, 'prediction': filename})
    engine.write_csv(out/'predictions.csv', rows)
    engine.write_json(out/'provenance.json', {'checkpoint_sha256': file_hash(checkpoint),
        'baseline_sha256': file_hash(baseline_checkpoint), 'manifest_sha256': file_hash(manifest),
        'note': 'No labels, FOVs, distance targets or oracle hypothesis selection used.'})
    return out


def probe_region_shape(cfg, baseline_checkpoint, mechanism, output, steps=100, samples=2,
                        learning_rate=.001):
    """Training-only fitting diagnostic; never emits a formal model checkpoint."""
    if steps < 1 or samples < 1 or learning_rate <= 0:
        raise ValueError('Invalid probe steps/samples/learning rate')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    engine.set_seed(cfg.training.seed)
    records = read_manifest(cfg.data.train_manifest)[:samples]
    model, device, _ = build_system(cfg, baseline_checkpoint, mechanism)
    # No random augmentation; measure repeatable small-set fitting first.
    loader = engine.make_loader(records, model.image_size, cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    enabled, dtype, scaler = amp_settings(cfg, device)
    out.mkdir(parents=True)
    initial, _ = score(model, loader, cfg, device)
    iterator = iter(loader)
    history = []
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        losses, gradients = update(model, batch, device, optimizer, scaler, enabled, dtype)
        history.append({'step': step+1, 'losses': losses, 'gradient_norms': gradients})
    final, _ = score(model, loader, cfg, device, out)
    engine.write_json(out/'probe.json', {'initial_training': initial, 'final_training': final,
        'history': history, 'steps': steps, 'samples': len(records), 'learning_rate': learning_rate,
        'mechanism': mechanism.to_dict(), 'baseline_sha256': file_hash(baseline_checkpoint),
        'scope': 'Training-only learnability diagnostic, not validation or performance evidence.'})
    return out
