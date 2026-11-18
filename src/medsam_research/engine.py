"""Training, native-resolution evaluation, and label-free inference entry points."""

import csv
import json
import random
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from .config import SAM2_COMMIT
from .data import (
    BinaryDataset,
    audit_manifests,
    collate_samples,
    file_hash,
    image_tensor,
    load_pair,
    read_manifest,
)
from .metrics import evaluate_loader, native_prediction, segmentation_loss
from .model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use device: cpu for debugging only.")
    return device


def make_loader(records, image_size, cfg, shuffle=False, augment=False):
    return DataLoader(
        BinaryDataset(records, image_size, cfg.data.mask_threshold,
                      augmentation="light" if augment else "none"),
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_samples,
        pin_memory=cfg.training.device.startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_revision():
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_dirty():
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def adapter_state(model):
    return {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }


def load_adapter(model, path, cfg):
    # Only load weights-only checkpoints emitted by this project, not arbitrary pickle objects.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or saved.get("sam2_commit") != SAM2_COMMIT:
        raise ValueError("Unsupported checkpoint format or SAM 2 version")
    if saved["model_config"] != {
        "sam_config": cfg.model.sam_config,
        "prompt_tokens": cfg.model.prompt_tokens,
    }:
        raise ValueError("Checkpoint and model configuration differ")
    if saved["pretrained_sha256"] != file_hash(cfg.model.checkpoint):
        raise ValueError("Pretrained checkpoint does not match the training run")
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(saved["adapter"]) != expected:
        raise ValueError("Unexpected or missing trainable weights")
    model.load_state_dict(saved["adapter"], strict=False)
    return saved


def train(cfg):
    t = cfg.training
    set_seed(t.seed)
    device = resolve_device(t.device)
    audit = audit_manifests(
        [cfg.data.train_manifest, cfg.data.val_manifest], cfg.data.mask_threshold
    )
    model = build_model(cfg.model, device)
    train_loader = make_loader(read_manifest(cfg.data.train_manifest), model.image_size, cfg,
                               True, augment=t.augmentation == "light")
    val_loader = make_loader(read_manifest(cfg.data.val_manifest), model.image_size, cfg)
    out = Path(t.output_dir)
    # Never silently reuse a previous experiment directory.
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    pretrained_sha = file_hash(cfg.model.checkpoint)
    versions = {}
    for name in ["torch", "torchvision", "SAM-2", "numpy", "Pillow", "scipy", "PyYAML"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    provenance = {
        "project_commit": git_revision(),
        "project_dirty": git_dirty(),
        "sam2_commit": SAM2_COMMIT,
        "pretrained_sha256": pretrained_sha,
        "versions": versions,
        "data_audit": audit,
        "seed": t.seed,
        "device": str(device),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "evaluation_grid": "original_image",
        "boundary_tolerance_pixels": t.boundary_tolerance,
        "augmentation": t.augmentation,
        "threshold": t.threshold,
    }
    if device.type == "cuda":
        provenance["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    write_json(out / "provenance.json", provenance)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=t.learning_rate,
        weight_decay=t.weight_decay,
    )
    use_amp = bool(t.amp and device.type == "cuda")
    # fp16 overflows in the mask decoder on some GPUs; bf16 needs no loss scaling.
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    best, started = -1.0, time.perf_counter()
    for epoch in range(1, t.epochs + 1):
        model.train()
        loss_sum, samples = 0.0, 0
        for batch in train_loader:
            images, targets = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss = segmentation_loss(logits, targets, batch["valid"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(images)
            samples += len(images)
        scores, rows = evaluate_loader(model, val_loader, device, t.threshold, t.boundary_tolerance)
        record = {"epoch": epoch, "train_loss": loss_sum / samples, "validation": scores}
        print(json.dumps(record), flush=True)
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = {
            "format_version": 1,
            "sam2_commit": SAM2_COMMIT,
            "pretrained_sha256": pretrained_sha,
            "model_config": {
                "sam_config": cfg.model.sam_config,
                "prompt_tokens": cfg.model.prompt_tokens,
            },
            "epoch": epoch,
            "validation": scores,
            "adapter": adapter_state(model),
            "threshold": t.threshold,
            "mask_threshold": cfg.data.mask_threshold,
            "boundary_tolerance": t.boundary_tolerance,
        }
        torch.save(checkpoint, out / "last.pt")
        if scores["dice"] > best:
            best = scores["dice"]
            torch.save(checkpoint, out / "best.pt")
            write_csv(out / "best_validation.csv", rows)
    timing = {"wall_seconds": time.perf_counter() - started, "best_validation_dice": best}
    if device.type == "cuda":
        timing["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    write_json(out / "training_summary.json", timing)
    return out


def _restore(cfg, checkpoint):
    device = resolve_device(cfg.training.device)
    model = build_model(cfg.model, device)
    saved = load_adapter(model, checkpoint, cfg)
    # Evaluation settings are fixed in the training artifact, not tuned on test labels.
    if (
        cfg.training.threshold != saved["threshold"]
        or cfg.data.mask_threshold != saved["mask_threshold"]
        or cfg.training.boundary_tolerance != saved["boundary_tolerance"]
    ):
        raise ValueError("Evaluation threshold/mask convention differs from training")
    return model.eval(), device


def evaluate(cfg, checkpoint, manifest, output):
    # Both source manifests remain available to detect test overlap before scoring.
    report = audit_manifests(
        [cfg.data.train_manifest, cfg.data.val_manifest, manifest], cfg.data.mask_threshold
    )
    model, device = _restore(cfg, checkpoint)
    loader = make_loader(read_manifest(manifest), model.image_size, cfg)
    summary, rows = evaluate_loader(
        model, loader, device, cfg.training.threshold, cfg.training.boundary_tolerance
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "summary.json",
        {
            **summary,
            "data_audit": report,
            "checkpoint_sha256": file_hash(checkpoint),
            "threshold": cfg.training.threshold,
            "boundary_tolerance_pixels": cfg.training.boundary_tolerance,
        },
    )
    write_csv(out / "per_image.csv", rows)
    print(json.dumps(summary), flush=True)
    return summary


@torch.no_grad()
def predict(cfg, checkpoint, manifest, output):
    records = read_manifest(manifest, require_masks=False)
    # Ignore any mask column: label-free inference must not read annotations.
    from dataclasses import replace

    model, device = _restore(cfg, checkpoint)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    mapping = []
    for index, record in enumerate(records):
        image, _ = load_pair(replace(record, mask=None))
        tensor = image_tensor(image, model.image_size)[None].to(device)
        logits = model(tensor)
        mask = native_prediction(logits, (image.height, image.width), cfg.training.threshold)
        filename = f"{index:06d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / filename)
        mapping.append({"id": record.sample_id, "image": str(record.image), "prediction": filename})
    write_csv(out / "predictions.csv", mapping)
    write_json(
        out / "provenance.json",
        {
            "checkpoint_sha256": file_hash(checkpoint),
            "manifest_sha256": file_hash(manifest),
            "threshold": cfg.training.threshold,
            "sam2_commit": SAM2_COMMIT,
        },
    )
    return out
