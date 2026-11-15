"""Per-image binary metrics on the original image grid (macro averaged)."""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.nn import functional as F


def segmentation_loss(logits, target, valid=None):
    if valid is None:
        valid = torch.ones_like(target)
    if valid.shape != target.shape or (valid.flatten(1).sum(1) == 0).any():
        raise ValueError("Each training sample needs a nonempty valid-pixel mask")
    weights = valid.float().flatten(1)
    errors = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    bce = ((errors.flatten(1) * weights).sum(1) / weights.sum(1)).mean()
    prob = logits.float().sigmoid().flatten(1) * weights
    target = target.float().flatten(1) * weights
    dice = (2 * (prob * target).sum(1) + 1) / (prob.sum(1) + target.sum(1) + 1)
    return 0.5 * bce + 0.5 * (1 - dice).mean()


def native_prediction(logits, shape, threshold):
    logits = F.interpolate(logits.float(), size=shape, mode="bilinear", align_corners=False)
    return logits[0, 0].sigmoid().ge(threshold).cpu().numpy()


def binary_metrics(prediction, target, tolerance=2, valid=None):
    p, y = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if p.ndim != 2 or p.shape != y.shape:
        raise ValueError("Expected equally sized 2D masks")
    if tolerance < 0:
        raise ValueError("Negative boundary tolerance")
    valid = np.ones_like(y) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != y.shape or not valid.any():
        raise ValueError("Invalid or empty evaluation region")
    p, y = p & valid, y & valid
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
        "foreground_fraction": ys / int(valid.sum()),
        "predicted_foreground_fraction": ps / int(valid.sum()),
        "evaluation_pixels": int(valid.sum()),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=0.5, boundary_tolerance=2):
    model.eval()
    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        for index, (sid, target) in enumerate(zip(batch["id"], batch["native_mask"])):
            prediction = native_prediction(logits[index : index + 1], target.shape, threshold)
            rows.append(
                {
                    "id": sid,
                    "fov_used": batch["fov_used"][index],
                    **binary_metrics(
                        prediction, target, boundary_tolerance, batch["native_valid"][index]
                    ),
                }
            )
    if not rows:
        raise ValueError("Empty evaluation loader")
    summary = {
        key: float(np.mean([r[key] for r in rows])) for key in ("dice", "iou", "boundary_f1")
    }
    summary["samples"] = len(rows)
    summary["fov_samples"] = sum(r["fov_used"] for r in rows)
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
