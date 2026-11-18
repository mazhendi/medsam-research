"""Two-stage text-localization/repair training, diagnostics, and fixed label-free inference."""

import json
import math
import time
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .data import audit_manifests, file_hash, image_tensor, load_pair, read_manifest
from .diagnostics import error_overlay, vessel_metrics
from .engine import _restore, git_revision, make_loader, set_seed, write_csv, write_json
from .metrics import binary_metrics, native_prediction, segmentation_loss
from .text_evidence import FrozenBiomedCLIP, read_prompt_map, resolve_prompts
from .text_verification import (
    REPAIR_MODES,
    DirectionalRepair,
    TextLocalizer,
    complementarity,
    repair_objective,
)


class TextSystem(nn.Module):
    def __init__(self, baseline, evidence, channels, width=16, repair_mode=None, threshold=.5):
        super().__init__()
        self.baseline, self.evidence = baseline, evidence
        baseline.requires_grad_(False)
        evidence.requires_grad_(False)
        self.image_size = baseline.image_size
        self.localizer = TextLocalizer(channels, evidence.text_dim, width)
        self.repair = None
        if repair_mode is not None:
            self.localizer.requires_grad_(False)
            self.repair = DirectionalRepair(channels, width, threshold, repair_mode)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        self.evidence.eval()
        if self.repair is not None:
            self.localizer.eval()
        return self

    def forward(self, images, prompts, null_text=False, apply_repair=True):
        with torch.no_grad():
            visual, pyramid = self.baseline.forward_with_pyramid(images)
            similarity, text = self.evidence(images, prompts)
        prior = self.localizer(pyramid, visual, similarity, text)
        result = {'visual': visual, 'text': prior, 'pyramid': pyramid, 'similarity': similarity}
        if null_text:
            result['text_null'] = self.localizer(pyramid, visual, torch.zeros_like(similarity),
                                                  torch.zeros_like(text))
        if self.repair is not None and apply_repair:
            repaired = self.repair(pyramid, visual, prior)
            result.update(repaired=repaired['logits'], repair_fields=repaired)
        return result


def _native_probability(logits, shape):
    return F.interpolate(logits.float(), shape, mode='bilinear', align_corners=False)[0, 0].sigmoid()


def _save_probability(path, array):
    Image.fromarray(np.round(np.clip(array, 0, 1)*255).astype(np.uint8)).save(path)


@torch.no_grad()
def score_system(system, loader, records, cfg, device, prompt, mapping, output=None):
    system.eval()
    rows, complements, repairs, images = [], [], [], []
    record_map = {r.sample_id: r for r in records}
    counter = 0
    for batch in loader:
        prompts = resolve_prompts(batch['id'], prompt, mapping)
        result = system(batch['image'].to(device), prompts, null_text=True)
        modes = [name for name in ('visual', 'text', 'text_null', 'repaired') if name in result]
        for i, sid in enumerate(batch['id']):
            target, valid = batch['native_mask'][i], batch['native_valid'][i]
            record = record_map[sid]
            predictions = {}
            image = load_pair(record, cfg.data.mask_threshold)[0] if output is not None else None
            for mode in modes:
                p = native_prediction(result[mode][i:i+1], target.shape, cfg.training.threshold)
                predictions[mode] = p
                rows.append({'id': sid, 'dataset': record.dataset, 'group': record.group, 'mode': mode,
                             **binary_metrics(p, target, cfg.training.boundary_tolerance, valid),
                             **vessel_metrics(p, target, valid)})
                if output is not None:
                    prefix = f'{counter:06d}_{mode}'
                    Image.fromarray((p & valid.astype(bool)).astype(np.uint8)*255).save(
                        output/f'{prefix}_mask.png')
                    error_overlay(image, p, target, valid).save(output/f'{prefix}_errors.png')
                    probability = _native_probability(result[mode][i:i+1], target.shape).cpu().numpy()
                    _save_probability(output/f'{prefix}_probability.png', probability)
            identity = {'id': sid, 'dataset': record.dataset, 'group': record.group}
            difference = (_native_probability(result['text'][i:i+1], target.shape)
                          - _native_probability(result['text_null'][i:i+1], target.shape)).abs()
            v = torch.as_tensor(valid.astype(bool), device=difference.device)
            complements.append({**identity, **complementarity(predictions['text'],
                                                              predictions['visual'], target, valid),
                                'text_null_probability_l1': difference[v].mean().item(),
                                'text_null_mask_disagreement': float(
                                    (predictions['text'] != predictions['text_null'])[valid.astype(bool)].mean())})
            if 'repaired' in result:
                y, v = target.astype(bool), valid.astype(bool)
                before, after = predictions['text'], predictions['repaired']
                repairs.append({**identity,
                    'recovered_fn_pixels': int((y & ~before & after & v).sum()),
                    'removed_fp_pixels': int((~y & before & ~after & v).sum()),
                    'new_fn_pixels': int((y & before & ~after & v).sum()),
                    'new_fp_pixels': int((~y & ~before & after & v).sum())})
                if output is not None and system.repair.mode not in ('residual', 'balanced-residual'):
                    fields = result['repair_fields']
                    domain = fields['domain'][i:i+1]
                    for label, value in (
                        ('recovery', (~domain)*fields['recovery_logits'][i:i+1].sigmoid()),
                        ('suppression', domain*fields['suppression_logits'][i:i+1].sigmoid()),
                    ):
                        native = F.interpolate(value.float(), target.shape, mode='bilinear',
                                               align_corners=False)[0, 0].cpu().numpy()
                        _save_probability(output/f'{counter:06d}_{label}_probability.png', native)
            if output is not None:
                response = F.interpolate(result['similarity'][i:i+1].float(), target.shape,
                                         mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                _save_probability(output/f'{counter:06d}_patch_cosine.png', (response+1)/2)
                images.append({'index': counter, 'id': sid, 'prompt': prompts[i],
                               'image': str(record.image)})
            counter += 1
    summary = {}
    keys = ('dice', 'iou', 'boundary_f1', 'precision', 'recall', 'cldice',
            'thin_skeleton_recall', 'medium_skeleton_recall', 'thick_skeleton_recall')
    for mode in dict.fromkeys(r['mode'] for r in rows):
        selected = [r for r in rows if r['mode'] == mode]
        summary[mode] = {}
        for key in keys:
            values = [r[key] for r in selected if r[key] is not None]
            summary[mode][key] = float(np.mean(values)) if values else None
    if output is not None:
        write_csv(output/'per_image.csv', rows)
        write_csv(output/'complementarity.csv', complements)
        fn_total = sum(r['text_fn_pixels'] for r in complements)
        fp_total = sum(r['text_fp_pixels'] for r in complements)
        write_json(output/'diagnostics.json', {
            'text_fn_pixels': fn_total, 'text_fp_pixels': fp_total,
            'visual_recovery_fraction_pooled': sum(r['visual_recovers_text_fn_pixels']
                for r in complements)/fn_total if fn_total else None,
            'visual_rejection_fraction_pooled': sum(r['visual_rejects_text_fp_pixels']
                for r in complements)/fp_total if fp_total else None,
            'text_null_probability_l1_mean': float(np.mean(
                [r['text_null_probability_l1'] for r in complements])),
            'text_null_mask_disagreement_mean': float(np.mean(
                [r['text_null_mask_disagreement'] for r in complements])),
            'note': 'Complementarity is annotation-relative diagnostic headroom. '
                    'Text-null is an ablation, not a calibrated reliability measure.'})
        write_csv(output/'images.csv', images)
        if repairs:
            write_csv(output/'repair_transitions.csv', repairs)
    return summary


def _load_artifact(path, kind):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1 or saved.get('kind') != kind:
        raise ValueError(f'Expected {kind} artifact')
    return saved


def _check_binding(saved, cfg, checkpoint, evidence):
    provenance = saved['provenance']
    if provenance['baseline_sha256'] != file_hash(checkpoint):
        raise ValueError('Text model was trained against a different baseline')
    if provenance['biomedclip'] != evidence.identity:
        raise ValueError('BiomedCLIP identity differs from training')
    previous = provenance['config']
    if (previous['model']['sam_config'] != cfg.model.sam_config
            or previous['model']['prompt_tokens'] != cfg.model.prompt_tokens
            or previous['data']['mask_threshold'] != cfg.data.mask_threshold
            or previous['training']['threshold'] != cfg.training.threshold
            or previous['training']['boundary_tolerance'] != cfg.training.boundary_tolerance):
        raise ValueError('Text evaluation settings differ from training')


def build_system(cfg, checkpoint, biomedclip_dir, localizer_path=None, repair_path=None, width=16):
    baseline, device = _restore(cfg, checkpoint)
    evidence = FrozenBiomedCLIP(biomedclip_dir, device)
    local = _load_artifact(localizer_path, 'text_localizer') if localizer_path else None
    repair = _load_artifact(repair_path, 'text_repair') if repair_path else None
    if repair is not None and local is None:
        raise ValueError('A localizer checkpoint is required for repair')
    if local:
        _check_binding(local, cfg, checkpoint, evidence)
        channels, width = local['provenance']['channels'], local['provenance']['width']
    else:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, baseline.image_size, baseline.image_size, device=device)
            _, pyramid = baseline.forward_with_pyramid(dummy)
            channels = [p.shape[1] for p in pyramid]
    mode = repair['provenance']['repair_mode'] if repair else None
    system = TextSystem(baseline, evidence, channels, width, mode, cfg.training.threshold).to(device)
    if local:
        system.localizer.load_state_dict(local['state'], strict=True)
    if repair:
        _check_binding(repair, cfg, checkpoint, evidence)
        if repair['provenance']['localizer_sha256'] != file_hash(localizer_path):
            raise ValueError('Repair is bound to another localizer checkpoint')
        system.repair.load_state_dict(repair['state'], strict=True)
    return system, device, local, repair


def train_text_model(cfg, checkpoint, biomedclip_dir, output, stage='localizer', localizer_path=None,
                     prompt='retinal blood vessels', prompt_map=None, epochs=30, learning_rate=.001,
                     width=16, repair_mode='full', direction_weight=.5, consistency_weight=.1):
    if stage not in ('localizer', 'repair') or epochs < 1 or width < 1:
        raise ValueError('Invalid stage, epochs or width')
    if repair_mode not in REPAIR_MODES:
        raise ValueError('Invalid repair mode')
    if (not math.isfinite(learning_rate) or learning_rate <= 0
            or not all(math.isfinite(v) and v >= 0 for v in (direction_weight, consistency_weight))):
        raise ValueError('Invalid learning rate or loss weights')
    if (stage == 'repair') != (localizer_path is not None):
        raise ValueError('Only repair training requires a localizer checkpoint')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    set_seed(cfg.training.seed)
    audit = audit_manifests([cfg.data.train_manifest, cfg.data.val_manifest], cfg.data.mask_threshold)
    system, device, local, _ = build_system(cfg, checkpoint, biomedclip_dir, localizer_path,
                                          width=width)
    if stage == 'repair':
        if local['epoch'] == 0:
            print('Diagnostic: localizer selected epoch 0 (identity); repair alone cannot establish text benefit.', flush=True)
        prompt = local['provenance']['prompt']
        mapping = local['provenance']['prompt_map']
        width = local['provenance']['width']
        system.localizer.requires_grad_(False)
        channels = local['provenance']['channels']
        system.repair = DirectionalRepair(channels, width, cfg.training.threshold, repair_mode).to(device)
    else:
        mapping = read_prompt_map(prompt_map)
        resolve_prompts(['validate'], prompt, mapping)
        channels = [layer.in_channels for layer in system.localizer.projections]
    head = system.localizer if stage == 'localizer' else system.repair
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate,
                                  weight_decay=cfg.training.weight_decay)
    train_records, val_records = (read_manifest(p) for p in
                                 (cfg.data.train_manifest, cfg.data.val_manifest))
    train_loader = make_loader(train_records, system.image_size, cfg, True)
    fit_loader = make_loader(train_records, system.image_size, cfg)
    val_loader = make_loader(val_records, system.image_size, cfg)
    generator = torch.Generator().manual_seed(cfg.training.seed)
    versions = {}
    for name in ('torch', 'open_clip_torch', 'timm', 'transformers', 'huggingface-hub'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'not installed (fixture only)'
    provenance = {'project_commit': git_revision(), 'stage': stage, 'versions': versions,
                  'baseline_sha256': file_hash(checkpoint), 'biomedclip': system.evidence.identity,
                  'config': cfg.to_dict(), 'data_audit': audit, 'prompt': prompt, 'prompt_map': mapping,
                  'channels': channels, 'width': width, 'repair_mode': repair_mode if stage == 'repair' else None,
                  'localizer_sha256': file_hash(localizer_path) if localizer_path else None,
                  'epochs': epochs, 'learning_rate': learning_rate, 'training_precision': 'FP32',
                  'direction_weight': direction_weight, 'consistency_weight': consistency_weight,
                  'trainable_parameters': sum(p.numel() for p in system.parameters() if p.requires_grad),
                  'preprocess': 'SAM square warp; BiomedCLIP same field resized to 224, official mean/std',
                  'patch_readout': 'contextualized ViT patches projected by pretrained image head; cosine',
                  'note': 'Research prototype, not Simple-ViLMedSAM IPP/BID; no claimed novelty/accuracy. '
                          'Stage 2 freezes both upstream branches. No label-driven inference.'}
    out.mkdir(parents=True)
    write_json(out/'provenance.json', provenance)
    mode = 'text' if stage == 'localizer' else 'repaired'
    initial = score_system(system, val_loader, val_records, cfg, device, prompt, mapping)
    write_json(out/'initial_validation.json', initial)
    best = initial[mode]['dice']
    kind = 'text_localizer' if stage == 'localizer' else 'text_repair'
    def artifact(epoch):
        return {'format_version': 1, 'kind': kind, 'epoch': epoch, 'provenance': provenance,
                'state': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    torch.save(artifact(0), out/f'best_{stage}.pt')
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs+1):
        system.train()
        losses, norms = [], []
        for batch in train_loader:
            texts = resolve_prompts(batch['id'], prompt, mapping)
            optimizer.zero_grad(set_to_none=True)
            result = system(batch['image'].to(device), texts, apply_repair=False)
            y, valid = batch['mask'].to(device), batch['valid'].to(device)
            if stage == 'localizer':
                loss = segmentation_loss(result['text'], y, valid)
            else:
                loss = repair_objective(head, result['pyramid'], result['visual'], result['text'],
                                        y, valid, repair_mode, direction_weight, consistency_weight,
                                        generator)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite text model loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            norms.append(norm.item())
            optimizer.step()
            losses.append((loss.item(), len(y)))
        training = score_system(system, fit_loader, train_records, cfg, device, prompt, mapping)
        validation = score_system(system, val_loader, val_records, cfg, device, prompt, mapping)
        record = {'epoch': epoch, 'train_loss': sum(value*n for value, n in losses)/sum(n for _, n in losses),
                  'training': training, 'validation': validation,
                  'gradient_norm_before_clip': float(np.mean(norms))}
        print(json.dumps(record), flush=True)
        with (out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        if validation[mode]['dice'] > best:
            best = validation[mode]['dice']
            torch.save(artifact(epoch), out/f'best_{stage}.pt')
        torch.save(artifact(epoch), out/f'last_{stage}.pt')
    saved = _load_artifact(out/f'best_{stage}.pt', kind)
    head.load_state_dict(saved['state'])
    if saved['epoch'] == 0:
        print('Diagnostic: validation selected identity initialization; no improvement established.', flush=True)
    scores = score_system(system, val_loader, val_records, cfg, device, prompt, mapping, out)
    timing = {'wall_seconds': time.perf_counter()-started}
    if device.type == 'cuda':
        timing['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    write_json(out/'summary.json', {'best_epoch': saved['epoch'], 'selected_identity': saved['epoch'] == 0,
                                   'modes': scores,
                                   'timing': timing, 'provenance': provenance})
    return scores


def evaluate_text_model(cfg, checkpoint, biomedclip_dir, localizer_path, manifest, output,
                        repair_path=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    paths = [cfg.data.train_manifest, cfg.data.val_manifest]
    if Path(manifest).resolve() != Path(cfg.data.val_manifest).resolve():
        paths.append(manifest)
    audit = audit_manifests(paths, cfg.data.mask_threshold)
    system, device, local, repair = build_system(cfg, checkpoint, biomedclip_dir, localizer_path,
                                                repair_path)
    records = read_manifest(manifest)
    loader = make_loader(records, system.image_size, cfg)
    provenance = local['provenance']
    out.mkdir(parents=True)
    scores = score_system(system, loader, records, cfg, device, provenance['prompt'],
                          provenance['prompt_map'], out)
    write_json(out/'summary.json', {'modes': scores, 'data_audit': audit,
                                   'localizer_sha256': file_hash(localizer_path),
                                   'repair_sha256': file_hash(repair_path) if repair else None,
                                   'note': 'Fixed weights; no updates or interventions.'})
    return scores


@torch.no_grad()
def predict_text_model(cfg, checkpoint, biomedclip_dir, localizer_path, manifest, output,
                       repair_path=None):
    records = read_manifest(manifest, require_masks=False)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    system, device, local, repair = build_system(cfg, checkpoint, biomedclip_dir, localizer_path,
                                                repair_path)
    system.eval()
    provenance = local['provenance']
    out.mkdir(parents=True)
    rows = []
    for index, record in enumerate(records):
        image, _ = load_pair(replace(record, mask=None))  # never reads mask/FOV columns
        texts = resolve_prompts([record.sample_id], provenance['prompt'], provenance['prompt_map'])
        result = system(image_tensor(image, system.image_size)[None].to(device), texts)
        mode = 'repaired' if repair else 'text'
        mask = native_prediction(result[mode], (image.height, image.width), cfg.training.threshold)
        filename = f'{index:06d}.png'
        Image.fromarray(mask.astype(np.uint8)*255).save(out/filename)
        rows.append({'id': record.sample_id, 'image': str(record.image), 'prompt': texts[0],
                     'prediction': filename})
    write_csv(out/'predictions.csv', rows)
    write_json(out/'provenance.json', {'manifest_sha256': file_hash(manifest),
                                     'localizer_sha256': file_hash(localizer_path),
                                     'repair_sha256': file_hash(repair_path) if repair else None,
                                     'baseline_sha256': file_hash(checkpoint),
                                     'biomedclip': system.evidence.identity,
                                     'threshold': cfg.training.threshold})
    return out
