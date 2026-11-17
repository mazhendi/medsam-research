"""Validation diagnostics and a fixed, label-independent native-crop control."""

import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torch.nn import functional as F

from .data import audit_manifests, file_hash, image_tensor, load_fov, load_pair, read_manifest
from .engine import _restore, git_revision, write_csv, write_json
from .metrics import binary_metrics


def vessel_metrics(prediction, target, valid, thin=4.0, medium=8.0):
    """Widths are 2*EDT at target skeleton pixels, in native pixels, not physical units."""
    if not 0 < thin < medium:
        raise ValueError("Require 0 < thin width < medium width")
    valid = np.asarray(valid, dtype=bool)
    p, y = np.asarray(prediction, dtype=bool) & valid, np.asarray(target, dtype=bool) & valid
    sp, sy = skeletonize(p), skeletonize(y)
    tp = int((p & y).sum())
    precision = tp / int(p.sum()) if p.any() else float(not y.any())
    recall = tp / int(y.sum()) if y.any() else float(not p.any())
    top_precision = float(y[sp].mean()) if sp.any() else 0.0
    top_recall = float(p[sy].mean()) if sy.any() else 0.0
    denom = top_precision + top_recall
    result = {
        "precision": precision, "recall": recall,
        "cldice": 2 * top_precision * top_recall / denom if denom else float(not (p | y).any()),
    }
    width = 2 * distance_transform_edt(y)
    for name, band in (
        ("thin", width <= thin),
        ("medium", (width > thin) & (width <= medium)),
        ("thick", width > medium),
    ):
        selected = sy & band
        count = int(selected.sum())
        result[f"{name}_skeleton_pixels"] = count
        result[f"{name}_skeleton_recall"] = float(p[selected].mean()) if count else None
    return result


def crop_boxes(width, height, grid=2, overlap=0.25):
    """Grid x grid fixed windows, covering native image without reading annotations."""
    if not 1 <= grid <= 8 or not 0 <= overlap < 1:
        raise ValueError("Require grid in [1, 8] and overlap in [0, 1)")
    def axis(length):
        size = min(length, math.ceil(length / (grid - (grid - 1) * overlap)))
        starts = sorted(set(round(v) for v in np.linspace(0, length - size, grid)))
        return [(s, s + size) for s in starts]
    return [(x0, y0, x1, y1) for y0, y1 in axis(height) for x0, x1 in axis(width)]


@torch.inference_mode()
def native_logits(model, image, device):
    logits = model(image_tensor(image, model.image_size)[None].to(device))
    return F.interpolate(
        logits.float(), (image.height, image.width), mode="bilinear", align_corners=False
    )[0, 0].cpu()


@torch.inference_mode()
def reread_logits(model, image, device, grid=2, overlap=0.25):
    boxes = crop_boxes(image.width, image.height, grid, overlap)
    total = torch.zeros(image.height, image.width)
    weights = torch.zeros_like(total)
    for box in boxes:
        x0, y0, x1, y1 = box
        patch = native_logits(model, image.crop(box), device)
        # Nonzero borders guarantee coverage even at image edges and with zero overlap.
        wy = torch.hann_window(y1 - y0, periodic=False).clamp_min(0.01)
        wx = torch.hann_window(x1 - x0, periodic=False).clamp_min(0.01)
        weight = wy[:, None] * wx[None, :]
        total[y0:y1, x0:x1] += patch * weight
        weights[y0:y1, x0:x1] += weight
    if (weights <= 0).any():
        raise RuntimeError("Uncovered pixels in crop assembly")
    return total / weights, boxes


def error_overlay(image, prediction, target, valid):
    """Red: false negative; blue: false positive; green: true positive; outside FOV dimmed."""
    valid = np.asarray(valid, dtype=bool)
    p, y = np.asarray(prediction, dtype=bool) & valid, np.asarray(target, dtype=bool) & valid
    rgb = np.array(image, dtype=np.float32)
    rgb[~valid] *= 0.25
    for pixels, color in ((y & ~p, [255, 0, 0]), (p & ~y, [0, 100, 255]), (p & y, [0, 255, 0])):
        rgb[pixels] = 0.35 * rgb[pixels] + 0.65 * np.array(color)
    return Image.fromarray(rgb.astype(np.uint8))


def _timed(device, operation):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def diagnose(cfg, checkpoint, manifest, output, grid=2, overlap=0.25, blend=0.5,
             thin=4.0, medium=8.0):
    # This development command deliberately requires the configured validation manifest.
    if Path(manifest).resolve() != Path(cfg.data.val_manifest).resolve():
        raise ValueError("diagnose requires the configured validation manifest; keep test locked")
    crop_boxes(10, 10, grid, overlap)
    if not 0 <= blend <= 1 or not 0 < thin < medium:
        raise ValueError("Invalid blend or vessel width bands")
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    audit = audit_manifests([cfg.data.train_manifest, manifest], cfg.data.mask_threshold)
    model, device = _restore(cfg, checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    rows, mappings = [], []
    for index, record in enumerate(read_manifest(manifest)):
        image, target = load_pair(record, cfg.data.mask_threshold)
        valid = load_fov(record, target.shape)
        # Warm up once outside timing. Each inference uses batch=1 and FP32, as evaluate does.
        if index == 0:
            native_logits(model, image, device)
        global_logits, global_seconds = _timed(device, lambda: native_logits(model, image, device))
        (patch_logits, boxes), patch_seconds = _timed(
            device, lambda: reread_logits(model, image, device, grid, overlap)
        )
        combined = (1 - blend) * global_logits + blend * patch_logits
        for mode, logits, seconds, calls in (
            ("global", global_logits, global_seconds, 1),
            ("reread", combined, global_seconds + patch_seconds, 1 + len(boxes)),
        ):
            prediction = logits.sigmoid().ge(cfg.training.threshold).numpy()
            prefix = f"{index:06d}_{mode}"
            Image.fromarray((prediction & valid.astype(bool)).astype(np.uint8) * 255).save(
                out / f"{prefix}_mask.png"
            )
            error_overlay(image, prediction, target, valid).save(out / f"{prefix}_errors.png")
            rows.append({
                "id": record.sample_id, "dataset": record.dataset, "group": record.group,
                "mode": mode, "seconds": seconds, "model_calls": calls,
                "fov_used": record.fov is not None,
                **binary_metrics(prediction, target, cfg.training.boundary_tolerance, valid),
                **vessel_metrics(prediction, target, valid, thin, medium),
            })
        mappings.append({"index": index, "id": record.sample_id, "image": str(record.image),
                         "crop_boxes_xyxy": boxes})
        print(f"Diagnosed {record.sample_id}", flush=True)
    keys = ["dice", "iou", "boundary_f1", "precision", "recall", "cldice",
            "thin_skeleton_recall", "medium_skeleton_recall", "thick_skeleton_recall",
            "seconds", "model_calls"]
    summaries = {}
    for mode in ("global", "reread"):
        selected = [r for r in rows if r["mode"] == mode]
        summaries[mode] = {}
        for key in keys:
            values = [r[key] for r in selected if r[key] is not None]
            summaries[mode][key] = float(np.mean(values)) if values else None
    paired = []
    for before, after in zip(rows[::2], rows[1::2]):
        paired.append({"id": before["id"], "dataset": before["dataset"], "group": before["group"],
                       **{f"delta_{k}": after[k] - before[k]
                          if after[k] is not None and before[k] is not None else None
                          for k in keys}})
    grouped = {}
    for row in paired:
        if row["group"]:
            grouped.setdefault((row["dataset"], row["group"]), []).append(row)
    group_rows = []
    for (dataset, group), members in grouped.items():
        group_row = {"dataset": dataset, "group": group, "images": len(members)}
        for key in keys:
            values = [r[f"delta_{key}"] for r in members if r[f"delta_{key}"] is not None]
            group_row[f"delta_{key}"] = float(np.mean(values)) if values else None
        group_rows.append(group_row)
    if group_rows:
        write_csv(out / "per_group_deltas.csv", group_rows)
    write_csv(out / "per_image.csv", rows)
    write_csv(out / "paired_deltas.csv", paired)
    write_json(out / "images.json", mappings)
    write_json(out / "summary.json", {
        "samples": len(mappings), "known_groups": len(group_rows),
        "modes": summaries, "data_audit": audit,
        "checkpoint_sha256": file_hash(checkpoint), "project_commit": git_revision(),
        "grid": grid, "overlap": overlap, "global_patch_logit_blend": blend,
        "thin_max_width_pixels": thin, "medium_max_width_pixels": medium,
        "threshold": cfg.training.threshold,
        "boundary_tolerance_pixels": cfg.training.boundary_tolerance,
        "note": "Validation-only fixed-grid control; no updates, no label-based crop selection. "
                "Image macro means; paired eyes are not independent subjects. "
                "Timing includes resize/transfers/stitching; reread reuses global pass; "
                "clDice is a centerline overlap metric, not proof of connected anatomy.",
    })
    return summaries
