"""Training-only paired local positives and strictly GT-excluded background crops."""

import random

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import load_pair
from .sam2_unet_workflow import structure_loss


def positive_box(mask, rng):
    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Local-positive sampling requires a nonempty training GT")
    left, right, top, bottom = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    factor = rng.uniform(1.2, 2.5)
    crop_width = min(width, max(1, round((right - left) * factor)))
    crop_height = min(height, max(1, round((bottom - top) * factor)))
    # Random placement while preserving the entire annotated foreground bbox.
    x = rng.randint(max(0, right - crop_width), min(left, width - crop_width))
    y = rng.randint(max(0, bottom - crop_height), min(top, height - crop_height))
    return x, y, x + crop_width, y + crop_height


def background_boxes(mask, rng, count=4, attempts=80, margin_fraction=.01):
    height, width = mask.shape
    margin = max(2, round(min(height, width) * margin_fraction))
    protected = maximum_filter(mask.astype(np.uint8), size=2 * margin + 1) > 0
    integral = np.pad(protected, ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1)
    boxes = []
    for _ in range(attempts):
        cw = max(1, round(width * rng.uniform(.15, .4)))
        ch = max(1, round(height * rng.uniform(.15, .4)))
        x, y = rng.randint(0, width - cw), rng.randint(0, height - ch)
        box = (x, y, x + cw, y + ch)
        occupied = (integral[y + ch, x + cw] - integral[y, x + cw]
                    - integral[y + ch, x] + integral[y, x])
        if box not in boxes and occupied == 0:
            boxes.append(box)
            if len(boxes) == count:
                break
    # If any valid crops exist, repetition maintains the fixed compute budget.
    # If none exist, caller explicitly falls back to FULL IMAGE + ORIGINAL LABEL.
    return [boxes[index % len(boxes)] for index in range(count)] if boxes else []


def tensor_pair(image, mask, rng, size=352):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    mask = mask.resize((size, size), Image.Resampling.BICUBIC)
    if rng.random() < .5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < .5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    x = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
    x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor(
        [.229, .224, .225])[:, None, None]
    return x, torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]


class LocalBackgroundDataset(Dataset):
    def __init__(self, records, seed=42, candidates=4, attempts=80, margin_fraction=.01):
        self.records, self.seed, self.epoch = records, seed, 0
        self.candidates, self.attempts, self.margin_fraction = candidates, attempts, margin_fraction

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # Coordinates independent of selection mode and DataLoader worker scheduling.
        rng = random.Random(self.seed + self.epoch * 1000003 + index * 9973)
        image, mask = load_pair(self.records[index])
        label = Image.fromarray(mask * 255)
        full, full_gt = tensor_pair(image, label, rng)
        box = positive_box(mask, rng)
        positive, positive_gt = tensor_pair(image.crop(box), label.crop(box), rng)
        if positive_gt.max() < .5:
            positive, positive_gt = full.clone(), full_gt.clone()
        boxes = background_boxes(mask, rng, self.candidates, self.attempts, self.margin_fraction)
        candidates = []
        for box in boxes:
            x, y = tensor_pair(image.crop(box), label.crop(box), rng)
            if y.count_nonzero():
                raise RuntimeError("Background sampler crossed training foreground")
            candidates.append(x)
        available = bool(candidates)
        if not available:
            candidates = [full.clone() for _ in range(self.candidates)]
        return dict(full=full, full_gt=full_gt, positive=positive, positive_gt=positive_gt,
                    candidates=torch.stack(candidates), available=available,
                    background_gt=torch.zeros_like(full_gt) if available else full_gt)


@torch.no_grad()
def select_background(model, candidates, mode, use_hard):
    """Score every candidate in ALL arms, using one physical batch at a time."""
    was_training = model.training
    model.eval()
    try:
        scores = torch.stack([model(candidates[:, index]).sigmoid().mean((1, 2, 3))
                              for index in range(candidates.shape[1])], 1)
    finally:
        model.train(was_training)
    indices = scores.argmax(1) if mode == "hard" and use_hard else torch.zeros(
        candidates.shape[0], dtype=torch.long, device=candidates.device)
    selected = candidates[torch.arange(len(indices), device=candidates.device), indices]
    return selected, scores.gather(1, indices[:, None]).squeeze(1)


def background_loss(logits, target, available):
    """BCE for real negatives; normal structure loss for explicit full-image fallbacks."""
    batch = len(logits)
    total = logits.sum() * 0
    if available.any():
        if target[available].count_nonzero():
            raise ValueError("Negative examples must have strictly zero masks")
        total = total + F.binary_cross_entropy_with_logits(
            logits[available], target[available]) * available.sum() / batch
    fallback = ~available
    if fallback.any():
        total = total + structure_loss(logits[fallback], target[fallback]) * fallback.sum() / batch
    return total


def local_training_step(model, batch, optimizer, mode, use_hard):
    """One update, K no-grad forwards and three sequential grad forwards in every arm."""
    device = next(model.parameters()).device
    batch = {name: value.to(device) for name, value in batch.items()}
    selected, score = select_background(model, batch["candidates"], mode, use_hard)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if mode == "matched_full":
        views = [(batch["full"], batch["full_gt"], False)] * 3
    else:
        views = [(batch["full"], batch["full_gt"], False),
                 (batch["positive"], batch["positive_gt"], False),
                 (selected, batch["background_gt"], True)]
    losses = []
    for image, target, negative in views:
        logits = model(image)
        loss = background_loss(logits, target, batch["available"]) if negative else structure_loss(
            logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite local-background training loss")
        (loss / 3).backward()
        losses.append(loss.detach().item())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    valid = batch["available"]
    score_sum = float(score[valid].sum())
    count = int(valid.sum())
    return dict(loss=sum(losses) / 3, full_loss=losses[0], positive_loss=losses[1],
                background_loss=losses[2], selected_background_score=score_sum / max(1, count),
                background_score_sum=score_sum, background_count=count,
                examples=len(batch["full"]), background_fallbacks=int((~batch["available"]).sum()),
                forward_calls=batch["candidates"].shape[1] + 3, backward_calls=3)
