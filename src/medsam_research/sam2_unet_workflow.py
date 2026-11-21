"""Isolated SAM2-UNet baseline; no dependency on the historical SAM decoder workflow."""

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import load_pair, read_manifest


def structure_loss(logits, target):
    weight = 1 + 5 * (F.avg_pool2d(target, 31, 1, 15) - target).abs()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (weight * bce).sum((2, 3)) / weight.sum((2, 3))
    probability = logits.sigmoid()
    intersection = (probability * target * weight).sum((2, 3))
    union = ((probability + target) * weight).sum((2, 3))
    return (bce + 1 - (intersection + 1) / (union - intersection + 1)).mean()


def scaled_size(base, rate):
    if rate == 1:
        return base
    return max(32, int(round(base * rate / 32)) * 32)


class PolypUNetDataset(Dataset):
    def __init__(self, records, size=352, augment=True):
        self.records, self.size, self.augment = records, size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = load_pair(self.records[index])
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        # Upstream SAM2-UNet resizes grayscale masks with bicubic (soft targets).
        mask = Image.fromarray(mask * 255).resize(
            (self.size, self.size), Image.Resampling.BICUBIC
        )
        if self.augment and random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augment and random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        x = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor(
            [.229, .224, .225]
        )[:, None, None]
        y = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        return x, y


def merge_training_manifests(manifests, output, expected_count=1450):
    """Combine train/development rows, refusing duplicate IDs or image pixels."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    records = [record for path in manifests for record in read_manifest(path)]
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} training pairs, found {len(records)}")
    ids, hashes = set(), set()
    for record in records:
        image, _ = load_pair(record)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if record.sample_id in ids or digest in hashes:
            raise ValueError(f"Duplicate training ID or image: {record.sample_id}")
        ids.add(record.sample_id)
        hashes.add(digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "mask", "dataset", "group"])
        writer.writeheader()
        for record in records:
            writer.writerow(dict(id=record.sample_id, image=str(record.image),
                                 mask=str(record.mask), dataset=record.dataset, group=record.group))
    return {"output": str(output), "count": len(records),
            "datasets": dict(Counter(record.dataset for record in records))}


def load_unet_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("architecture") != "sam2_unet_hiera_l":
        raise ValueError("Expected architecture: sam2_unet_hiera_l (original SAM2 weights)")
    for key in ("train_manifest", "encoder_checkpoint", "output"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg.get("image_size", 352) != 352:
        raise ValueError("This baseline protocol requires base image_size=352")
    if cfg.get("epochs", 20) < 1 or cfg.get("batch_size", 16) < 1:
        raise ValueError("epochs and batch_size must be positive")
    scales = cfg.get("scales", [1.0])
    if not scales or any(rate <= 0 for rate in scales):
        raise ValueError("scales must contain positive rates")
    return cfg


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def train_unet(config_path):
    from .sam2_unet import SAM2UNet

    cfg = load_unet_config(config_path)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(cfg["encoder_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download original SAM2 sam2_hiera_large.pt: {checkpoint}")
    device = torch.device(cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; full Hiera-L training requires the user's GPU")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records = read_manifest(cfg["train_manifest"])
    if len(records) != cfg.get("expected_train_count", 1450):
        raise ValueError(f"Expected 1450 training pairs, found {len(records)}")
    # Fail before training on broken masks or mismatched native image/mask sizes.
    for record in records:
        load_pair(record)
    model = SAM2UNet(checkpoint).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("learning_rate", .001), weight_decay=cfg.get("weight_decay", .0005),
    )
    epochs = cfg.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7)
    loader = DataLoader(
        PolypUNetDataset(records), batch_size=cfg.get("batch_size", 16), shuffle=True,
        num_workers=cfg.get("workers", 4), pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
    )
    output.mkdir(parents=True)
    provenance = {
        "config": cfg, "train_count": len(records), "selection": "fixed_final_epoch",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "manifest_sha256": hashlib.sha256(Path(cfg["train_manifest"]).read_bytes()).hexdigest(),
        "upstream": "WZH0120/SAM2-UNet@01598e5e9912ffb23f965ecbebf4d1dfecbaa56e",
        "precision": "fp32", "mask_resize": "bicubic_soft_target",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.
        updates = 0
        lr = optimizer.param_groups[0]["lr"]
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            for rate in cfg.get("scales", [1.0]):
                size = scaled_size(352, rate)
                x, y = images, target
                if size != 352:
                    x = F.interpolate(images, (size, size), mode="bilinear", align_corners=True)
                    y = F.interpolate(target, (size, size), mode="bilinear", align_corners=True)
                optimizer.zero_grad(set_to_none=True)
                loss = sum(structure_loss(pred, y) for pred in model(x))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, optimizer step {steps}")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                updates += 1
                steps += 1
        scheduler.step()
        row = dict(epoch=epoch, loss=loss_sum / updates, learning_rate=lr,
                   optimizer_steps=steps, updates=updates)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch % cfg.get("save_every", 5) == 0 or epoch == epochs:
            destination = output / ("last.pt" if epoch == epochs else f"epoch_{epoch:03d}.pt")
            torch.save(dict(architecture=cfg["architecture"], epoch=epoch,
                            model=model.state_dict(), optimizer_steps=steps,
                            provenance=provenance), destination)
    return {"output": str(output), "checkpoint": str(output / "last.pt"), "steps": steps}


@torch.inference_mode()
def predict_unet(config_path, checkpoint_path, manifest, output, normalization="raw"):
    from .sam2_unet import SAM2UNet

    cfg = load_unet_config(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if normalization not in {"raw", "per-image"}:
        raise ValueError("normalization must be raw or per-image")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("architecture") != cfg["architecture"]:
        raise ValueError("Not a project SAM2-UNet checkpoint")
    device = torch.device(cfg.get("device", "cuda"))
    model = SAM2UNet()
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    records = read_manifest(manifest, require_masks=False)
    output.mkdir(parents=True)
    mapping = []
    for index, record in enumerate(records):
        with Image.open(record.image) as im:
            image = im.convert("RGB")
        x = image.resize((352, 352), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(x, dtype=np.float32) / 255).permute(2, 0, 1)
        x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor(
            [.229, .224, .225]
        )[:, None, None]
        logits = model(x[None].to(device))[0]
        probability = F.interpolate(logits, (image.height, image.width), mode="bilinear",
                                    align_corners=False).sigmoid()[0, 0].cpu().numpy()
        if normalization == "per-image":
            probability = (probability - probability.min()) / (
                probability.max() - probability.min() + 1e-8
            )
        filename = f"{index:06d}.png"
        # Upstream uses truncation, not rounding. Both exports record this explicitly.
        Image.fromarray((probability * 255).astype(np.uint8)).save(output / filename)
        mapping.append({"id": record.sample_id, "image": str(record.image),
                        "prediction": filename})
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "prediction"])
        writer.writeheader()
        writer.writerows(mapping)
    (output / "provenance.json").write_text(json.dumps(dict(
        architecture=cfg["architecture"], checkpoint=str(Path(checkpoint_path).resolve()),
        epoch=state["epoch"], normalization=normalization, encoding="uint8_floor",
        native_grid=True, inference_size=352, count=len(records),
    ), indent=2))
    return {"output": str(output), "count": len(records)}
