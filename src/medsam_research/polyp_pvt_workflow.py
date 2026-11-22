"""Polyp-PVT main baseline: no SAM dependency and no test-set model selection."""
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import file_hash, load_pair, read_manifest
from .polyp_pvt import SOURCE_HASHES, UPSTREAM_COMMIT, build_polyp_pvt
from .sam2_unet_workflow import scaled_size, seed_worker, structure_loss


def image_tensor(image):
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1)
    return (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor(
        [.229, .224, .225])[:, None, None]


class PolypPVTDataset(Dataset):
    """Released default: bilinear RGB/mask resize, no random augmentation."""
    def __init__(self, records, size=352):
        self.records, self.size = records, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = load_pair(self.records[index])
        size = (self.size, self.size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = Image.fromarray(mask * 255).resize(size, Image.Resampling.BILINEAR)
        target = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255)[None]
        return image_tensor(image), target


def load_pvt_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("architecture") != "polyp_pvt_b2":
        raise ValueError("Expected architecture: polyp_pvt_b2")
    for key in ("upstream", "encoder_checkpoint", "train_manifest", "output"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg.get("image_size") != 352 or cfg.get("augmentation") != "none":
        raise ValueError("Released baseline requires image_size=352 and augmentation=none")
    for key in ("epochs", "batch_size", "expected_train_count"):
        if not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["batch_size"] < 2:
        raise ValueError("Training batch_size must be >=2 for official BatchNorm branches")
    if not cfg.get("scales") or any(v <= 0 for v in cfg["scales"]):
        raise ValueError("scales must be nonempty and positive")
    if cfg.get("learning_rate", 0) <= 0 or cfg.get("gradient_clip", 0) <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    return cfg


def train_pvt(config_path):
    cfg = load_pvt_config(config_path)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run full training on the user's GPU")
    if not Path(cfg["encoder_checkpoint"]).is_file():
        raise FileNotFoundError(f"Download official PVTv2-B2 ImageNet weights: {cfg['encoder_checkpoint']}")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records = read_manifest(cfg["train_manifest"])
    if len(records) != cfg["expected_train_count"]:
        raise ValueError(f"Expected {cfg['expected_train_count']} train pairs, found {len(records)}")
    seen = set()
    for record in records:
        load_pair(record)
        digest = file_hash(record.image)
        if digest in seen:
            raise ValueError(f"Duplicate training image: {record.image}")
        seen.add(digest)
    model = build_polyp_pvt(cfg["upstream"], cfg["encoder_checkpoint"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    loader = DataLoader(PolypPVTDataset(records), batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=cfg.get("workers", 4),
                        pin_memory=device.type == "cuda", drop_last=True,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    if not len(loader):
        raise ValueError("Training pool is smaller than batch_size")
    provenance = dict(config=cfg, upstream_commit=UPSTREAM_COMMIT, source_hashes=SOURCE_HASHES,
                      selection="fixed_final_epoch_no_test_access", precision="fp32",
                      train_count=len(records), drop_last=True, learning_rate_schedule="constant",
                      manifest_sha256=file_hash(cfg["train_manifest"]),
                      encoder_sha256=file_hash(cfg["encoder_checkpoint"]),
                      trainable_parameters=sum(p.numel() for p in model.parameters()),
                      prediction="sum_of_two_logits", mask_resize="bilinear_soft_target")
    output.mkdir(parents=True)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    steps = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, updates = 0., 0
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            for rate in cfg["scales"]:
                size = scaled_size(352, rate)
                x, y = images, target
                if rate != 1:
                    x = F.interpolate(x, (size, size), mode="bilinear", align_corners=True)
                    y = F.interpolate(y, (size, size), mode="bilinear", align_corners=True)
                optimizer.zero_grad(set_to_none=True)
                loss = sum(structure_loss(logits, y) for logits in model(x))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss at epoch {epoch}, step {steps}")
                loss.backward()
                # Upstream clips each gradient element, not the global gradient norm.
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg["gradient_clip"])
                optimizer.step()
                steps += 1
                updates += 1
                total += loss.item()
        row = dict(epoch=epoch, loss=total / updates, optimizer_steps=steps,
                   learning_rate=optimizer.param_groups[0]["lr"], updates=updates)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch == cfg["epochs"]:
            torch.save(dict(architecture=cfg["architecture"], model=model.state_dict(),
                            epoch=epoch, optimizer_steps=steps, provenance=provenance),
                       output / "last.pt")
    return dict(output=str(output), checkpoint=str(output / "last.pt"), steps=steps)


def validate_inference_size(size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 64 or size % 32:
        raise ValueError("inference_size must be an integer multiple of 32, at least 64")
    return size


@torch.inference_mode()
def predict_pvt(config_path, checkpoint, manifest, output, normalization="raw",
                inference_size=352):
    validate_inference_size(inference_size)
    cfg = load_pvt_config(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if normalization not in {"raw", "per-image"}:
        raise ValueError("Expected raw or per-image normalization")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("architecture") != cfg["architecture"]:
        raise ValueError("Expected project Polyp-PVT checkpoint")
    model = build_polyp_pvt(cfg["upstream"])
    model.load_state_dict(state["model"], strict=True)
    device = torch.device(cfg.get("device", "cuda"))
    model.to(device).eval()
    records = read_manifest(manifest, require_masks=False)
    output.mkdir(parents=True)
    mapping = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for index, record in enumerate(records):
        with Image.open(record.image) as handle:
            image = handle.convert("RGB")
        x = image_tensor(image.resize((inference_size, inference_size), Image.Resampling.BILINEAR))
        first, second = model(x[None].to(device))
        logits = first + second
        probability = F.interpolate(logits, (image.height, image.width), mode="bilinear",
                                    align_corners=False).sigmoid()[0, 0].cpu().numpy()
        if normalization == "per-image":
            probability = (probability - probability.min()) / (
                probability.max() - probability.min() + 1e-8)
        filename = f"{index:06d}.png"
        Image.fromarray((probability * 255).astype(np.uint8)).save(output / filename)
        mapping.append(dict(id=record.sample_id, image=str(record.image), prediction=filename))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "prediction"])
        writer.writeheader()
        writer.writerows(mapping)
    (output / "provenance.json").write_text(json.dumps(dict(
        architecture=cfg["architecture"], checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        manifest_sha256=file_hash(manifest), normalization=normalization, encoding="uint8_floor",
        prediction="sigmoid(upsample(logit1+logit2))", epoch=state["epoch"],
        upstream_commit=UPSTREAM_COMMIT, native_grid=True, count=len(records),
        inference_size=inference_size, training_image_size=state.get("provenance", {}).get(
            "config", {}).get("image_size"), device=str(device), precision="fp32",
        elapsed_seconds=elapsed, seconds_per_image=elapsed / len(records) if records else None,
        peak_cuda_allocated_bytes=peak_memory,
        timing_scope="image loading, preprocessing, forward, native resize, PNG save; includes warmup"
    ), indent=2))
    return dict(output=str(output), count=len(records), inference_size=inference_size)


POLYP_TEST_SETS = ("Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")


def sweep_pvt(config_path, checkpoint, manifest_dir, output, sizes=(352, 448, 512),
              normalization="raw", manifest_suffix="_dedup.csv"):
    """One checkpoint, independent resolutions; no GT access, ensembling or selection."""
    sizes = list(sizes)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be nonempty and distinct")
    for size in sizes:
        validate_inference_size(size)
    if normalization not in {"raw", "per-image"}:
        raise ValueError("Expected raw or per-image normalization")
    load_pvt_config(config_path)
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifests = {name: Path(manifest_dir) / (name + manifest_suffix) for name in POLYP_TEST_SETS}
    # Fail before exporting any predictions when one of the five manifests is absent.
    for path in manifests.values():
        records = read_manifest(path, require_masks=False)
        if not records:
            raise ValueError(f"Empty prediction manifest: {path}")
        for record in records:
            if not record.image.is_file():
                raise FileNotFoundError(record.image)
    checkpoint = Path(checkpoint).resolve()
    checkpoint_hash = file_hash(checkpoint)
    manifest_hashes = {name: file_hash(path) for name, path in manifests.items()}
    output.mkdir(parents=True)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  sizes=sizes, normalization=normalization, selection="none_diagnostic_only",
                  manifest_sha256=manifest_hashes, results=[])
    for size in sizes:
        for name, manifest in manifests.items():
            if file_hash(checkpoint) != checkpoint_hash or file_hash(manifest) != manifest_hashes[name]:
                raise RuntimeError("Checkpoint or manifest changed during sweep")
            print(f"Predicting {name} at {size} x {size}", flush=True)
            result = predict_pvt(config_path, checkpoint, manifest, output / str(size) / name,
                                 normalization=normalization, inference_size=size)
            report["results"].append(dict(dataset=name, **result))
            (output / "sweep.json").write_text(json.dumps(report, indent=2))
    return report
