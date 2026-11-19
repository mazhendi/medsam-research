"""Isolated train/evaluate/predict/probe workflow for the boundary-contrast prototype."""

import json
import time
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import engine
from .boundary_contrast import BoundaryContrastConfig, BoundaryContrastSAM2, boundary_contrast_loss
from .config import SAM2_COMMIT
from .data import audit_manifests, file_hash, image_tensor, load_pair, read_manifest
from .metrics import binary_metrics, native_prediction

KIND = "boundary_contrast_sam2_v2"
LEGACY_KIND = "boundary_contrast_sam2_v1"


def read_artifact(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") not in (KIND, LEGACY_KIND) or saved.get("format_version") != 1:
        raise ValueError("Expected a boundary-contrast checkpoint, not a baseline or probe")
    if saved.get("sam2_commit") != SAM2_COMMIT:
        raise ValueError("SAM2 source version mismatch")
    # Version-1 files used a different, randomly initialized custom decoder.
    # Make that architecture explicit; never silently reinterpret old weights.
    expected = "custom_decoder" if saved["kind"] == LEGACY_KIND else "baseline_adapter"
    previous = dict(saved["mechanism"])
    previous.setdefault("architecture", expected)
    if previous["architecture"] != expected:
        raise ValueError("Checkpoint kind and decoder architecture disagree")
    previous.setdefault("train_decoder", False)
    saved["mechanism"] = BoundaryContrastConfig(**previous).validate().to_dict()
    return saved


def build_system(cfg, baseline_checkpoint, mechanism=None, checkpoint=None):
    if not Path(baseline_checkpoint).is_file():
        raise FileNotFoundError(baseline_checkpoint)
    saved = read_artifact(checkpoint) if checkpoint else None
    if saved:
        if saved["baseline_sha256"] != file_hash(baseline_checkpoint):
            raise ValueError("Different baseline checkpoint from the training run")
        if saved["evaluation"] != {
            "threshold": cfg.training.threshold,
            "mask_threshold": cfg.data.mask_threshold,
            "boundary_tolerance": cfg.training.boundary_tolerance,
        }:
            raise ValueError("Evaluation settings differ from training")
        previous = saved["mechanism"]
        if mechanism is not None and mechanism.to_dict() != previous:
            raise ValueError("Mechanism settings differ from checkpoint")
        mechanism = BoundaryContrastConfig(**previous).validate()
    if mechanism is None:
        raise ValueError("Mechanism configuration is required")
    baseline, device = engine._restore(cfg, baseline_checkpoint)
    model = BoundaryContrastSAM2(baseline, mechanism).to(device)
    if saved:
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(saved["adapter"]) != expected:
            raise ValueError("Unexpected or missing mechanism/decoder parameters")
        # Shape errors raise in load_state_dict; missing frozen base parameters
        # are intentional, supplied by the bound baseline checkpoint.
        model.load_state_dict(saved["adapter"], strict=False)
    return model, device, saved


def artifact(model, cfg, baseline_checkpoint, epoch, scores, source_audit):
    return {
        "kind": KIND if model.config.architecture == "baseline_adapter" else LEGACY_KIND,
        "format_version": 1,
        "sam2_commit": SAM2_COMMIT,
        "baseline_sha256": file_hash(baseline_checkpoint),
        "mechanism": model.config.to_dict(),
        "adapter": engine.adapter_state(model),
        "epoch": epoch,
        "validation": scores,
        "source_config": cfg.to_dict(),
        "source_audit": source_audit,
        "evaluation": {
            "threshold": cfg.training.threshold,
            "mask_threshold": cfg.data.mask_threshold,
            "boundary_tolerance": cfg.training.boundary_tolerance,
        },
    }


def precision_recall(prediction, target, valid):
    p, y, v = prediction.astype(bool), target.astype(bool), valid.astype(bool)
    tp = int((p & y & v).sum())
    ps, ys = int((p & v).sum()), int((y & v).sum())
    return {
        "precision": tp / ps if ps else float(ys == 0),
        "recall": tp / ys if ys else float(ps == 0),
    }


@torch.no_grad()
def score(model, loader, cfg, device, output=None):
    model.eval()
    rows, diagnostic_rows, changes = [], [], []
    counter = 0
    for batch in loader:
        result = model.forward_details(batch["image"].to(device))
        for i, sid in enumerate(batch["id"]):
            y, v = batch["native_mask"][i], batch["native_valid"][i]
            predictions = {}
            for mode, key in [("baseline", "baseline_logits"), ("boundary_contrast", "logits")]:
                prediction = native_prediction(
                    result[key][i : i + 1], y.shape, cfg.training.threshold
                )
                predictions[mode] = prediction
                rows.append(
                    {
                        "id": sid,
                        "mode": mode,
                        **binary_metrics(prediction, y, cfg.training.boundary_tolerance, v),
                        **precision_recall(prediction, y, v),
                    }
                )
                if output:
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(
                        Path(output) / f"{counter:06d}_{mode}_mask.png"
                    )
            old, new = predictions["baseline"], predictions["boundary_contrast"]
            truth, region = y.astype(bool), v.astype(bool)
            corrected = (old != truth) & (new == truth) & region
            damaged = (old == truth) & (new != truth) & region
            changes.append(
                {
                    "id": sid,
                    "corrected_pixels": int(corrected.sum()),
                    "damaged_pixels": int(damaged.sum()),
                    "corrected_foreground": int((corrected & truth).sum()),
                    "corrected_background": int((corrected & ~truth).sum()),
                    "lost_foreground": int((damaged & truth).sum()),
                    "new_false_positive": int((damaged & ~truth).sum()),
                }
            )
            diagnostic_rows.append(
                {
                    "id": sid,
                    "mean_reliability": float(result["reliability"][i].float().mean()),
                    "feature_gates": (
                        result["feature_gates"].detach().cpu().tolist()
                        if "feature_gates" in result
                        else None
                    ),
                    "mean_contrast_gate": float(result["boundary_mix"][i].float().mean()),
                    "mean_detail_gates": [
                        float(g[i].float().mean()) for g in result["fusion_gates"]
                    ],
                }
            )
            if output:
                np.savez_compressed(
                    Path(output) / f"{counter:06d}_states.npz",
                    coarse_probability=result["probability"][i].float().cpu().numpy(),
                    normal=result["normal"][i].float().cpu().numpy(),
                    reliability=result["reliability"][i].float().cpu().numpy(),
                    contrast_gate=result["boundary_mix"][i].float().cpu().numpy(),
                )
            counter += 1
    if not rows:
        raise ValueError("Empty evaluation split")
    summary = {}
    for mode in ("baseline", "boundary_contrast"):
        selected = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            key: float(np.mean([r[key] for r in selected]))
            for key in (
                "dice",
                "iou",
                "boundary_f1",
                "precision",
                "recall",
                "predicted_foreground_fraction",
            )
        }
        summary[mode]["samples"] = len(selected)
    if output:
        engine.write_csv(Path(output) / "per_image.csv", rows)
        engine.write_csv(Path(output) / "paired_changes.csv", changes)
        engine.write_json(Path(output) / "states.json", diagnostic_rows)
    return summary, rows


def amp_settings(cfg, device):
    enabled = bool(cfg.training.amp and device.type == "cuda")
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled, dtype, torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16)


def update(model, batch, device, optimizer, scaler, enabled, dtype):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        result = model.forward_details(batch["image"].to(device), include_baseline=False)
        losses = boundary_contrast_loss(
            result, batch["mask"].to(device), batch["valid"].to(device), model.config
        )
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("Nonfinite boundary-contrast loss")
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    gradients = {}
    names = ["head", "head.boundary", "head.fusions", "head.coarse"]
    names.append(
        "head.output_projections"
        if model.config.architecture == "baseline_adapter"
        else "head.decoder"
    )
    if model.config.train_decoder:
        names.append("baseline.sam.sam_mask_decoder")
    for name in names:
        grad = [
            p.grad.detach().float().square().sum()
            for p in model.get_submodule(name).parameters()
            if p.grad is not None
        ]
        gradients[name] = float(torch.stack(grad).sum().sqrt()) if grad else 0.0
    if model.config.architecture == "baseline_adapter":
        g = model.head.alpha.grad
        gradients["head.alpha"] = float(g.float().norm()) if g is not None else 0.0
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
    )
    scaler.step(optimizer)
    scaler.update()
    return {name: value.detach().item() for name, value in losses.items()}, gradients


def train_boundary_contrast(cfg, baseline_checkpoint, mechanism, output, epochs=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    epochs = cfg.training.epochs if epochs is None else epochs
    if epochs < 1:
        raise ValueError("epochs must be positive")
    engine.set_seed(cfg.training.seed)
    audit = audit_manifests(
        [cfg.data.train_manifest, cfg.data.val_manifest], cfg.data.mask_threshold
    )
    model, device, _ = build_system(cfg, baseline_checkpoint, mechanism)
    training = engine.make_loader(
        read_manifest(cfg.data.train_manifest),
        model.image_size,
        cfg,
        True,
        augment=cfg.training.augmentation != "none",
    )
    validation = engine.make_loader(read_manifest(cfg.data.val_manifest), model.image_size, cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    enabled, dtype, scaler = amp_settings(cfg, device)
    out.mkdir(parents=True)
    versions = {}
    for package in ("torch", "SAM-2", "numpy", "scipy", "Pillow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    engine.write_json(
        out / "provenance.json",
        {
            "project_commit": engine.git_revision(),
            "project_dirty": engine.git_dirty(),
            "source_config": cfg.to_dict(),
            "mechanism": mechanism.to_dict(),
            "baseline_sha256": file_hash(baseline_checkpoint),
            "pretrained_sha256": file_hash(cfg.model.checkpoint),
            "data_audit": audit,
            "versions": versions,
            "epochs": epochs,
            "training_precision": str(dtype) if enabled else "float32",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_passes_per_image": 1,
            "custom_decoder_passes_per_image": int(mechanism.architecture == "custom_decoder"),
            "sam_decoder_passes_training": int(mechanism.architecture == "baseline_adapter"),
            "sam_decoder_passes_paired_evaluation": 2
            if mechanism.architecture == "baseline_adapter"
            else 1,
            "note": "Architecture is explicit. V2 preserves the original decoder through zero-initialized feature gates; no logit addition.",
        },
    )
    if mechanism.architecture == "baseline_adapter":
        model.eval()
        with torch.no_grad():
            images = next(iter(validation))["image"].to(device)
            check = model.forward_details(images)
            torch.testing.assert_close(
                check["logits"], check["baseline_logits"], atol=1e-6, rtol=1e-6
            )
            gap = float((check["logits"] - check["baseline_logits"]).abs().max())
        engine.write_json(
            out / "identity_check.json",
            {
                "max_logit_difference": gap,
                "passed": True,
                "scope": "First validation batch, before optimizer updates",
            },
        )
    initial, _ = score(model, validation, cfg, device)
    engine.write_json(out / "initial_validation.json", initial)
    best, best_epoch = initial["boundary_contrast"]["dice"], 0
    torch.save(
        artifact(model, cfg, baseline_checkpoint, 0, initial, audit),
        out / "best_boundary_contrast.pt",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        totals, norms, samples = {}, {}, 0
        for batch in training:
            loss, gradients = update(model, batch, device, optimizer, scaler, enabled, dtype)
            count = len(batch["id"])
            for name, value in loss.items():
                totals[name] = totals.get(name, 0.0) + value * count
            for name, value in gradients.items():
                norms[name] = norms.get(name, 0.0) + value * count
            samples += count
        scores, _ = score(model, validation, cfg, device)
        record = {
            "epoch": epoch,
            "losses": {k: v / samples for k, v in totals.items()},
            "gradient_norms": {k: v / samples for k, v in norms.items()},
            "validation": scores,
        }
        print(json.dumps(record, allow_nan=False), flush=True)
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        saved = artifact(model, cfg, baseline_checkpoint, epoch, scores, audit)
        torch.save(saved, out / "last_boundary_contrast.pt")
        if scores["boundary_contrast"]["dice"] > best:
            best, best_epoch = scores["boundary_contrast"]["dice"], epoch
            torch.save(saved, out / "best_boundary_contrast.pt")
    best_saved = read_artifact(out / "best_boundary_contrast.pt")
    model.load_state_dict(best_saved["adapter"], strict=False)
    scores, _ = score(model, validation, cfg, device, out)
    engine.write_json(
        out / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "validation": scores,
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "initial_head_selected": best_epoch == 0,
            "baseline_selected": best_epoch == 0 and mechanism.architecture == "baseline_adapter",
        },
    )
    if best_epoch == 0:
        print(
            "Validation selected epoch 0: no improvement over initialization established.",
            flush=True,
        )
    return out


def evaluate_boundary_contrast(cfg, baseline_checkpoint, checkpoint, manifest, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    saved = read_artifact(checkpoint)
    # Audit against manifests saved at training time, not replacement paths in a
    # current config, so changing configs cannot bypass held-out data checks.
    source = saved["source_config"]["data"]
    paths = list(
        dict.fromkeys(
            str(Path(p).resolve())
            for p in (source["train_manifest"], source["val_manifest"], manifest)
        )
    )
    audit = audit_manifests(paths, cfg.data.mask_threshold)
    current = {r["manifest"]: r for r in audit}
    for expected in saved["source_audit"]:
        actual = current[expected["manifest"]]
        if any(actual[key] != expected[key] for key in ("sha256", "decoded_data_sha256")):
            raise ValueError("Source training/validation data changed since training")
    model, device, _ = build_system(cfg, baseline_checkpoint, checkpoint=checkpoint)
    loader = engine.make_loader(read_manifest(manifest), model.image_size, cfg)
    out.mkdir(parents=True)
    scores, _ = score(model, loader, cfg, device, out)
    engine.write_json(
        out / "summary.json",
        {
            "modes": scores,
            "audit": audit,
            "checkpoint_sha256": file_hash(checkpoint),
            "baseline_sha256": file_hash(baseline_checkpoint),
        },
    )
    return scores


@torch.no_grad()
def predict_boundary_contrast(cfg, baseline_checkpoint, checkpoint, manifest, output):
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
        result = model.forward_details(tensor, include_baseline=False)
        mask = native_prediction(
            result["logits"], (image.height, image.width), cfg.training.threshold
        )
        filename = f"{i:06d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / filename)
        rows.append({"id": record.sample_id, "prediction": filename})
    engine.write_csv(out / "predictions.csv", rows)
    engine.write_json(
        out / "provenance.json",
        {
            "checkpoint_sha256": file_hash(checkpoint),
            "baseline_sha256": file_hash(baseline_checkpoint),
            "manifest_sha256": file_hash(manifest),
            "note": "No labels, FOVs or ground-truth boundary directions used for prediction.",
        },
    )
    return out


def probe_boundary_contrast(
    cfg, baseline_checkpoint, mechanism, output, steps=100, samples=2, learning_rate=0.001
):
    """Training-only fitting diagnostic; never emits a formal model checkpoint."""
    if steps < 1 or samples < 1 or learning_rate <= 0:
        raise ValueError("Invalid probe steps/samples/learning rate")
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    engine.set_seed(cfg.training.seed)
    records = read_manifest(cfg.data.train_manifest)[:samples]
    model, device, _ = build_system(cfg, baseline_checkpoint, mechanism)
    # No random augmentation; measure repeatable small-set fitting first.
    loader = engine.make_loader(records, model.image_size, cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
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
        history.append({"step": step + 1, "losses": losses, "gradient_norms": gradients})
    final, _ = score(model, loader, cfg, device, out)
    engine.write_json(
        out / "probe.json",
        {
            "initial_training": initial,
            "final_training": final,
            "history": history,
            "steps": steps,
            "samples": len(records),
            "learning_rate": learning_rate,
            "mechanism": mechanism.to_dict(),
            "baseline_sha256": file_hash(baseline_checkpoint),
            "scope": "Training-only learnability diagnostic, not validation or performance evidence.",
        },
    )
    return out
