"""Per-image binary metrics on the original image grid (macro averaged)."""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.nn import functional as F


def segmentation_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    prob = logits.float().sigmoid().flatten(1)
    target = target.float().flatten(1)
    dice = (2 * (prob * target).sum(1) + 1) / (prob.sum(1) + target.sum(1) + 1)
    return 0.5 * bce + 0.5 * (1 - dice).mean()


def native_prediction(logits, shape, threshold):
    logits = F.interpolate(logits.float(), size=shape, mode="bilinear", align_corners=False)
    return logits[0, 0].sigmoid().ge(threshold).cpu().numpy()


def binary_metrics(prediction, target, tolerance=2):
    p, y = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if p.ndim != 2 or p.shape != y.shape:
        raise ValueError("Expected equally sized 2D masks")
    if tolerance < 0:
        raise ValueError("Negative boundary tolerance")
    ps, ys = int(p.sum()), int(y.sum())
    inter = int((p & y).sum())
    total = ps + ys
    union = ps + ys - inter
    pb = p & ~binary_erosion(p, border_value=0)
    yb = y & ~binary_erosion(y, border_value=0)
    if not pb.any() or not yb.any():
        boundary_f1 = float(not pb.any() and not yb.any())
    else:
        precision = (distance_transform_edt(~yb)[pb] <= tolerance).mean()
        recall = (distance_transform_edt(~pb)[yb] <= tolerance).mean()
        boundary_f1 = (
            float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        )
    return {
        "dice": 2 * inter / total if total else 1.0,
        "iou": inter / union if union else 1.0,
        "boundary_f1": boundary_f1,
        "foreground_fraction": ys / y.size,
        "predicted_foreground_fraction": ps / p.size,
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=0.5, boundary_tolerance=2):
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        for index, (sid, target) in enumerate(zip(batch["id"], batch["native_mask"])):
            prediction = native_prediction(logits[index : index + 1], target.shape, threshold)
            rows.append({"id": sid, **binary_metrics(prediction, target, boundary_tolerance)})
    if not rows:
        raise ValueError("Empty evaluation loader")
    summary = {
        key: float(np.mean([r[key] for r in rows])) for key in ("dice", "iou", "boundary_f1")
    }
    summary["samples"] = len(rows)
    positive = [r for r in rows if r["foreground_fraction"] > 0]
    negative = [r for r in rows if r["foreground_fraction"] == 0]
    summary["positive_samples"] = len(positive)
    summary["empty_target_samples"] = len(negative)
    summary["positive_dice"] = float(np.mean([r["dice"] for r in positive])) if positive else None
    summary["empty_target_false_positive_rate"] = (
        float(np.mean([r["predicted_foreground_fraction"] > 0 for r in negative]))
        if negative
        else None
    )
    return summary, rows
