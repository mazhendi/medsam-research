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
from .polyp_pvt_augmentation import augment_pair, augmentation_policy
from .sam2_unet_workflow import scaled_size, seed_worker, structure_loss


def image_tensor(image):
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1)
    return (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor(
        [.229, .224, .225])[:, None, None]


class PolypPVTDataset(Dataset):
    """Legacy resize or opt-in paired geometry and RGB-only mild appearance."""
    def __init__(self, records, size=352, augmentation="none"):
        augmentation_policy(augmentation)
        self.records, self.size, self.augmentation = records, size, augmentation

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = load_pair(self.records[index])
        size = (self.size, self.size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = Image.fromarray(mask * 255).resize(size, Image.Resampling.BILINEAR)
        image, mask = augment_pair(image, mask, self.augmentation)
        target = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255)[None]
        return image_tensor(image), target


def uses_final_head_only(cfg):
    return (cfg["architecture"] in {"sds_adf_rtr_pvt_b2", "abc_pvt_b2"}
            or (cfg["architecture"] == "polyp_pvt_slots_b2"
                and cfg.get("slots", {}).get("deep_supervision", False)))


def load_pvt_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("architecture") not in {"polyp_pvt_b2", "subband_pvt_b2", "sds_adf_rtr_pvt_b2", "abc_pvt_b2", "polyp_pvt_slots_b2"}:
        raise ValueError("Unknown PVT architecture")
    if cfg["architecture"] == "polyp_pvt_slots_b2":
        from .polyp_pvt_slots import validate_slot_options

        validate_slot_options(cfg.get("slots", {}))
        import math
        weights = cfg.get("aux_weights", [.5, .5])
        if (not isinstance(weights, (list, tuple)) or len(weights) != 2
                or any(isinstance(w, bool) or not isinstance(w, (int, float))
                       or not math.isfinite(w) or w < 0 for w in weights)):
            raise ValueError("aux_weights must contain two finite nonnegative numbers")
        if cfg.get("image_size") != 352:
            raise ValueError("Interface-controlled training protocol is fixed at 352")
    if cfg["architecture"] == "abc_pvt_b2":
        from .abc_decoder import validate_abc_options

        validate_abc_options(cfg.get("decoder", {}))
        if cfg.get("image_size") != 352:
            raise ValueError("ABC training protocol is fixed at 352")
        if cfg.get("aux_weight", .5) < 0:
            raise ValueError("aux_weight must be nonnegative")
    if cfg["architecture"] == "sds_adf_rtr_pvt_b2":
        options = cfg.get("decoder", {})
        if not isinstance(options, dict) or set(options)-{"width", "sds", "adf", "rtr", "tau"}:
            raise ValueError("Invalid SDS/ADF/RTR decoder options")
        width = options.get("width", 64)
        if not isinstance(width, int) or isinstance(width, bool) or width < 8:
            raise ValueError("decoder width must be an integer >=8")
        if options.get("tau", .02) <= 0:
            raise ValueError("tau must be positive")
        if any(not isinstance(options.get(k, True), bool) for k in ("sds", "adf", "rtr")):
            raise ValueError("Module switches must be boolean")
        for key in ("coarse_weight", "intermediate_weight"):
            if cfg.get(key, .5 if key == "coarse_weight" else .25) < 0:
                raise ValueError("Supervision weights must be nonnegative")
        if cfg.get("image_size") != 352:
            raise ValueError("SDS/ADF/RTR training protocol is fixed at 352")
    if cfg["architecture"] == "subband_pvt_b2":
        options = cfg.get("decoder", {})
        if not isinstance(options, dict):
            raise ValueError("decoder must be a mapping")
        if set(options) - {"width", "attention_dim", "scsi", "bcsc", "boundary"}:
            raise ValueError("Unknown decoder option")
        width, dim = options.get("width", 64), options.get("attention_dim", 16)
        if not isinstance(width, int) or width < 16 or width % 8:
            raise ValueError("decoder width must be a multiple of 8 >=16")
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("attention_dim must be positive")
        if any(not isinstance(options.get(k, True), bool) for k in ("scsi", "bcsc", "boundary")):
            raise ValueError("Module switches must be boolean")
        if cfg.get("edge_weight", .2) < 0:
            raise ValueError("edge_weight must be nonnegative")
    for key in ("upstream", "encoder_checkpoint", "train_manifest", "output"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg.get("image_size") not in {352, 512} or isinstance(cfg.get("image_size"), bool):
        raise ValueError("Training image_size must be 352 or 512")
    augmentation_policy(cfg.get("augmentation"))
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


def build_configured_model(cfg, encoder_checkpoint=None):
    model = build_polyp_pvt(cfg["upstream"], encoder_checkpoint)
    if cfg["architecture"] == "subband_pvt_b2":
        from .subband_decoder import SubbandPVT

        # Only the encoder is retained; none of the original decoder modules are stored.
        model = SubbandPVT(model.backbone, **cfg.get("decoder", {}))
    elif cfg["architecture"] == "sds_adf_rtr_pvt_b2":
        from .scale_direction_decoder import ScaleDirectionPVT

        model = ScaleDirectionPVT(model.backbone, **cfg.get("decoder", {}))
    elif cfg["architecture"] == "abc_pvt_b2":
        from .abc_decoder import ABCPVT

        model = ABCPVT(model.backbone, **cfg.get("decoder", {}))
    elif cfg["architecture"] == "polyp_pvt_slots_b2":
        from .polyp_pvt_slots import PolypPVTSlots

        model = PolypPVTSlots(model, **cfg.get("slots", {}))
    return model


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
    model = build_configured_model(cfg, cfg["encoder_checkpoint"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    loader = DataLoader(PolypPVTDataset(records, size=cfg["image_size"],
                                             augmentation=cfg["augmentation"]),
                        batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=cfg.get("workers", 4),
                        pin_memory=device.type == "cuda", drop_last=True,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    if not len(loader):
        raise ValueError("Training pool is smaller than batch_size")
    provenance = dict(config=cfg, upstream_commit=UPSTREAM_COMMIT, source_hashes=SOURCE_HASHES,
                      augmentation_policy=augmentation_policy(cfg["augmentation"]),
                      selection="fixed_final_epoch_no_test_access", precision="fp32",
                      train_count=len(records), drop_last=True, learning_rate_schedule="constant",
                      manifest_sha256=file_hash(cfg["train_manifest"]),
                      encoder_sha256=file_hash(cfg["encoder_checkpoint"]),
                      trainable_parameters=sum(p.numel() for p in model.parameters()),
                      prediction=("final_head_only" if uses_final_head_only(cfg)
                                  else "sum_of_two_logits"), mask_resize="bilinear_soft_target",
                      training_sizes=[scaled_size(cfg["image_size"], r) for r in cfg["scales"]],
                      updates_per_epoch=len(loader) * len(cfg["scales"]))
    output.mkdir(parents=True)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    steps = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, updates = 0., 0
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            for rate in cfg["scales"]:
                size = scaled_size(cfg["image_size"], rate)
                x, y = images, target
                if rate != 1:
                    x = F.interpolate(x, (size, size), mode="bilinear", align_corners=True)
                    y = F.interpolate(y, (size, size), mode="bilinear", align_corners=True)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(x)
                if cfg["architecture"] == "subband_pvt_b2":
                    from .subband_decoder import decoder_loss

                    loss = decoder_loss(predictions, y, cfg.get("edge_weight", .2))
                elif cfg["architecture"] == "sds_adf_rtr_pvt_b2":
                    from .scale_direction_decoder import scale_direction_loss

                    loss = scale_direction_loss(predictions, y, cfg.get("coarse_weight", .5),
                                                cfg.get("intermediate_weight", .25))
                elif cfg["architecture"] == "abc_pvt_b2":
                    from .abc_decoder import abc_loss

                    loss = abc_loss(predictions, y, cfg.get("aux_weight", .5))
                elif cfg["architecture"] == "polyp_pvt_slots_b2":
                    from .polyp_pvt_slots import slots_loss

                    loss = slots_loss(predictions, y, cfg.get("aux_weights", [.5, .5]))
                else:
                    loss = sum(structure_loss(logits, y) for logits in predictions)
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
        if cfg["architecture"] in {"abc_pvt_b2", "polyp_pvt_slots_b2"}:
            row["scan_backends"] = scan_backend_counts(model)
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
        raise ValueError("Checkpoint architecture must match the supplied config")
    if cfg["architecture"] == "abc_pvt_b2":
        saved_options = state.get("provenance", {}).get("config", {}).get("decoder", {})
        for key in ("a_residual", "b_residual"):
            if cfg.get("decoder", {}).get(key, True) != saved_options.get(key, True):
                raise ValueError(f"Checkpoint/config mismatch: {key}; use its training config")
    if cfg["architecture"] == "polyp_pvt_slots_b2":
        saved = state.get("provenance", {}).get("config", {}).get("slots", {})
        for key, default in (("a", False), ("a_f2", False), ("deep_supervision", False), ("b", False), ("c", False),
                             ("a_residual", False), ("b_residual", False), ("cfm", True),
                             ("b_scan", "raster"), ("fusion", "aligned"), ("a1_variant", "original"), ("a2_variant", "original"), ("ablation", "none"), ("c_variant", "original")):
            if cfg.get("slots", {}).get(key, default) != saved.get(key, default):
                raise ValueError(f"Checkpoint/config mismatch: slots.{key}; use training config")
    model = build_configured_model(cfg)
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
        predictions = model(x[None].to(device))
        if uses_final_head_only(cfg):
            logits = predictions["final"]
        elif cfg["architecture"] == "subband_pvt_b2":
            logits = predictions["coarse"] + predictions["refined"]
        else:
            first, second = predictions
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
        prediction=("sigmoid(upsample(final_logit))" if uses_final_head_only(cfg)
                    else "sigmoid(upsample(logit1+logit2))"), epoch=state["epoch"],
        upstream_commit=UPSTREAM_COMMIT, native_grid=True, count=len(records),
        inference_size=inference_size, training_image_size=state.get("provenance", {}).get(
            "config", {}).get("image_size"), device=str(device), precision="fp32",
        elapsed_seconds=elapsed, seconds_per_image=elapsed / len(records) if records else None,
        peak_cuda_allocated_bytes=peak_memory,
        scan_backends=scan_backend_counts(model) if cfg["architecture"] in {"abc_pvt_b2", "polyp_pvt_slots_b2"} else None,
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


def scan_backend_counts(model):
    from collections import Counter

    return dict(Counter(m.last_backend for m in model.modules()
                        if getattr(m, "last_backend", None) is not None))


def probe_abc(config_path, steps=1, batch_size=None, image_size=None):
    """Synthetic full-resolution optimizer probe; writes no model or dataset files."""
    from .abc_decoder import abc_loss

    cfg = load_pvt_config(config_path)
    if cfg["architecture"] not in {"abc_pvt_b2", "polyp_pvt_slots_b2"}:
        raise ValueError("probe-abc-pvt requires an ABC or interface-controlled config")
    batch_size = cfg["batch_size"] if batch_size is None else batch_size
    image_size = max(scaled_size(352, rate) for rate in cfg["scales"]) if image_size is None else image_size
    validate_inference_size(image_size)
    if batch_size < 2 or steps < 1:
        raise ValueError("probe batch_size must be >=2 and steps >=1")
    device = torch.device(cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(cfg.get("seed", 42))
    model = build_configured_model(cfg, cfg["encoder_checkpoint"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    target = (torch.rand(batch_size, 1, image_size, image_size, device=device)>.7).float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        if cfg["architecture"] == "polyp_pvt_slots_b2":
            from .polyp_pvt_slots import slots_loss

            loss = slots_loss(predictions, target, cfg.get("aux_weights", [.5, .5]))
        else:
            loss = abc_loss(predictions, target, cfg.get("aux_weight", .5))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite probe loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Probe found missing or nonfinite gradients; use aux_weight>0")
        torch.nn.utils.clip_grad_value_(model.parameters(), cfg["gradient_clip"])
        optimizer.step()
        losses.append(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(architecture=cfg["architecture"], device=str(device), batch_size=batch_size,
                image_size=image_size, steps=steps, losses=losses,
                parameters=sum(p.numel() for p in model.parameters()),
                scan_backends=scan_backend_counts(model), elapsed_seconds=time.perf_counter()-started,
                peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                           if device.type == "cuda" else None),
                synthetic_probe=True, no_checkpoint_written=True)
