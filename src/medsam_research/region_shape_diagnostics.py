"""Label-based hypothesis diagnostics, never used by model.forward or prediction.

Oracle scores quantify a restricted family of fixed candidate masks. They are NOT
bounds on what the SAM decoder, retraining or a soft spatial mixture can achieve.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

from . import engine
from .data import audit_manifests, file_hash, read_manifest
from .metrics import native_prediction


def dice(prediction, target, valid):
    p, y = prediction.astype(bool) & valid, target.astype(bool) & valid
    total = int(p.sum()+y.sum())
    return 2*int((p & y).sum())/total if total else 1.


def window_oracle(candidates, target, valid, grid):
    """Maximize whole-image Dice over hard, nonoverlapping window selections.

    Dinkelbach fractional optimization: choices are separable given the current
    ratio. This avoids greedily maximizing per-window Dice, which need not maximize
    whole-image Dice. Ground truth is deliberately confined to this diagnostic.
    """
    k, height, width = candidates.shape
    gh, gw = grid
    if min(k, gh, gw) < 1 or gh > height or gw > width:
        raise ValueError('Invalid oracle window grid')
    candidates = candidates.astype(bool) & valid[None]
    y = target.astype(bool) & valid
    boxes, counts, overlaps = [], [], []
    for iy in range(gh):
        for ix in range(gw):
            ys = slice(iy*height//gh, (iy+1)*height//gh)
            xs = slice(ix*width//gw, (ix+1)*width//gw)
            window = candidates[:, ys, xs]
            boxes.append((ys, xs))
            counts.append(window.sum((1, 2)))
            overlaps.append((window & y[None, ys, xs]).sum((1, 2)))
    ps, tp = np.asarray(counts, float), np.asarray(overlaps, float)
    rows, ys_total = np.arange(len(boxes)), int(y.sum())
    ratio, selection = 0., np.zeros(len(boxes), dtype=int)
    if ys_total == 0:
        selection = ps.argmin(1)
    else:
        for _ in range(100):
            selection = (2*tp-ratio*ps).argmax(1)
            updated = 2*tp[rows, selection].sum()/(ps[rows, selection].sum()+ys_total)
            if abs(updated-ratio) < 1e-12:
                break
            ratio = updated
        else:
            raise RuntimeError('Window-oracle fractional optimization did not converge')
    mask = np.zeros_like(y)
    for choice, (ys, xs) in zip(selection, boxes):
        mask[ys, xs] = candidates[choice, ys, xs]
    return dice(mask, y, valid), mask, selection.reshape(gh, gw)


def error_auc(score, errors, valid):
    values, labels = score[valid].astype(float), errors[valid].astype(bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    ranks = rankdata(values, method='average')
    return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))


@torch.no_grad()
def hypothesis_report(occupancy, variance, spatial_weights, global_weights, final_mask,
                      target, valid, threshold, window_size):
    """occupancy, variance and weights are one sample, at the mechanism grid."""
    valid = np.asarray(valid, bool)
    target = np.asarray(target, bool)
    k, height, width = occupancy.shape
    if height % window_size or width % window_size:
        raise ValueError('Diagnostic window_size must divide the mechanism grid')
    native = F.interpolate(occupancy[None].float(), target.shape, mode='bilinear',
                           align_corners=False)[0].cpu().numpy()
    candidates = native >= threshold
    candidate_scores = [dice(mask, target, valid) for mask in candidates]
    ious, distances = [], []
    for i, j in combinations(range(k), 2):
        a, b = candidates[i] & valid, candidates[j] & valid
        union = int((a | b).sum())
        ious.append(int((a & b).sum())/union if union else 1.)
        distances.append(float(np.abs(native[i]-native[j])[valid].mean()))
    gh, gw = height//window_size, width//window_size
    # Very small native images cannot hold more nonempty tiles than pixels.
    grid = min(gh, target.shape[0]), min(gw, target.shape[1])
    oracle_dice, _, choices = window_oracle(candidates, target, valid, grid)
    v = F.interpolate(variance[None].float(), target.shape, mode='bilinear',
                       align_corners=False)[0, 0].cpu().numpy()
    global_consensus = (global_weights[:, None, None].float()*occupancy.float()).sum(0)
    consensus = (spatial_weights.float()*occupancy.float()).sum(0)

    def upsample_mask(tensor):
        return F.interpolate(tensor[None, None], target.shape, mode='bilinear',
                              align_corners=False)[0, 0].cpu().numpy() >= threshold

    result = {
        'candidate_dice': candidate_scores,
        'best_candidate_index_oracle': int(np.argmax(candidate_scores)),
        'best_candidate_dice_oracle': max(candidate_scores),
        'fixed_window_oracle_dice': oracle_dice,
        'window_oracle_gain_over_best_candidate': oracle_dice-max(candidate_scores),
        'window_oracle_choices': choices.tolist(),
        'oracle_grid': list(grid),
        'mean_pairwise_iou': float(np.mean(ious)) if ious else None,
        'mean_pairwise_probability_l1': float(np.mean(distances)) if distances else None,
        'global_consensus_dice': dice(upsample_mask(global_consensus), target, valid),
        'spatial_consensus_dice': dice(upsample_mask(consensus), target, valid),
        'final_dice': dice(final_mask, target, valid),
        'disagreement_error_auroc': error_auc(v, final_mask != target, valid),
        'mean_spatial_weight_std': float(spatial_weights.flatten(1).std(-1, unbiased=False).mean()),
    }
    return result


@torch.no_grad()
def diagnose_region_shape(cfg, baseline_checkpoint, checkpoint, manifest, output, window_size=8):
    from .region_shape_training import build_system, read_artifact

    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if type(window_size) is not int or window_size < 1:
        raise ValueError('window_size must be positive')
    saved = read_artifact(checkpoint)
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
    if model.config.grid_size % window_size:
        raise ValueError('window_size must divide mechanism grid_size')
    model.eval()
    loader = engine.make_loader(read_manifest(manifest), model.image_size, cfg)
    out.mkdir(parents=True)
    rows = []
    for batch in loader:
        result = model.forward_details(batch['image'].to(device))
        for i, sid in enumerate(batch['id']):
            y, v = batch['native_mask'][i], batch['native_valid'][i]
            final = native_prediction(result['logits'][i:i+1], y.shape, cfg.training.threshold)
            row = hypothesis_report(result['occupancy'][i], result['variance'][i],
                result['spatial_weights'][i], result['weights'][i], final, y, v,
                cfg.training.threshold, window_size)
            rows.append({'id': sid, **row})
    if not rows:
        raise ValueError('Empty diagnosis split')
    keys = ('best_candidate_dice_oracle', 'fixed_window_oracle_dice',
            'window_oracle_gain_over_best_candidate', 'mean_pairwise_iou',
            'mean_pairwise_probability_l1', 'global_consensus_dice', 'spatial_consensus_dice',
            'final_dice', 'disagreement_error_auroc', 'mean_spatial_weight_std')
    summary = {}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary['samples'] = len(rows)
    engine.write_json(out/'per_image.json', rows)
    engine.write_json(out/'summary.json', {'metrics': summary, 'audit': audit,
        'checkpoint_sha256': file_hash(checkpoint), 'baseline_sha256': file_hash(baseline_checkpoint),
        'window_size': window_size, 'threshold': cfg.training.threshold,
        'scope': 'Ground-truth-based diagnosis only. Oracle choices never enter inference; '
                 'fixed-candidate hard-selection score is NOT a bound on final decoder performance. '
                 'Use validation, not held-out test data, to choose model modifications.'})
    return summary
