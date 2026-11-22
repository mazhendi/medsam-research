"""Training/probing/export for the native-decoder research track only."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import load_pair, read_manifest
from .native_sam2 import build_native_model
from .sam2_unet_workflow import PolypUNetDataset, seed_worker, structure_loss


def load_native_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("architecture") != "native_sam2_hiera_l":
        raise ValueError("Expected native_sam2_hiera_l configuration, not SAM2-UNet")
    if cfg.get("image_size") != 352 or cfg.get("expected_train_count") != 1450:
        raise ValueError("This track requires base size 352 and a 1450-image training pool")
    if cfg.get("sam_config") != "configs/sam2/sam2_hiera_l.yaml":
        raise ValueError("This protocol requires the original SAM2 Hiera-L configuration")
    for key in ("train_manifest", "encoder_checkpoint", "output"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    modules = cfg.get("modules", {})
    if set(modules) != {"scds", "sbca", "rcsr"} or any(
        type(value) is not bool for value in modules.values()
    ):
        raise ValueError("modules must explicitly specify scds/sbca/rcsr booleans")
    if min(cfg.get("epochs", 20), cfg.get("batch_size", 16)) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if cfg.get("evidence_loss_weight", .1) < 0:
        raise ValueError("evidence_loss_weight cannot be negative")
    shrink = cfg.get("shrink_augmentation", {})
    if set(shrink) - {"probability", "min_scale", "max_scale"}:
        raise ValueError("Unknown shrink augmentation option")
    if (not 0 <= shrink.get("probability", 0.) <= 1
            or not 0 < shrink.get("min_scale", .5) <= shrink.get("max_scale", .8) <= 1):
        raise ValueError("Invalid shrink augmentation probability or scale range")
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def objective(model, images, target, cfg):
    logits = model(images)
    segmentation = structure_loss(logits, target)
    evidence = model.auxiliary_loss(target)
    loss = segmentation + cfg.get("evidence_loss_weight", .1) * evidence
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite native-SAM2 loss")
    return loss, segmentation, evidence


def _optimizer(model, cfg):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=cfg.get("learning_rate", .0001),
                             weight_decay=cfg.get("weight_decay", .0005))


def train_native(config_path):
    cfg = load_native_config(config_path)
    if "local_background" in cfg:
        raise ValueError("Use train-local-background for local-background configurations")
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    seed_all(cfg.get("seed", 42))
    records = read_manifest(cfg["train_manifest"])
    if len(records) != 1450:
        raise ValueError(f"Expected 1450 training pairs, found {len(records)}")
    for record in records:
        load_pair(record)
    device = torch.device(cfg.get("device", "cuda"))
    model = build_native_model(cfg, device)
    optimizer = _optimizer(model, cfg)
    epochs = cfg.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7)
    shrink = cfg.get("shrink_augmentation", {})
    dataset = PolypUNetDataset(records, shrink_probability=shrink.get("probability", 0.),
                               shrink_min=shrink.get("min_scale", .5),
                               shrink_max=shrink.get("max_scale", .8))
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 16),
                        shuffle=True, num_workers=cfg.get("workers", 4),
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(cfg.get("seed", 42)))
    output.mkdir(parents=True)
    provenance = dict(
        config=cfg, train_count=1450, selection="fixed_final_epoch", precision="fp32",
        manifest_sha256=hashlib.sha256(Path(cfg["train_manifest"]).read_bytes()).hexdigest(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        normalization="ImageNet", mask_resize="bicubic_soft_target",
        native_embedding_grid=[22, 22], native_high_resolution_grids=[[88, 88], [44, 44]],
        rcsr_location="after_first_native_transposed_conv_before_high_res_addition",
        shrink_augmentation=dict(probability=shrink.get("probability", 0.),
                                 min_scale=shrink.get("min_scale", .5),
                                 max_scale=shrink.get("max_scale", .8), image_padding="image_mean",
                                 mask_padding="background_zero", apply_to="training_only"),
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(3)
        lr = optimizer.param_groups[0]["lr"]
        for images, target in loader:
            optimizer.zero_grad(set_to_none=True)
            values = objective(model, images.to(device), target.to(device), cfg)
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            totals += [value.item() for value in values]
            steps += 1
        scheduler.step()
        row = dict(epoch=epoch, loss=totals[0] / len(loader),
                   segmentation_loss=totals[1] / len(loader), evidence_loss=totals[2] / len(loader),
                   learning_rate=lr, optimizer_steps=steps,
                   gains={name: float(p.detach().cpu()) for name, p in model.named_parameters()
                          if name.endswith("gain")})
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch % cfg.get("save_every", 5) == 0 or epoch == epochs:
            name = "last.pt" if epoch == epochs else f"epoch_{epoch:03d}.pt"
            torch.save(dict(architecture=cfg["architecture"], epoch=epoch,
                            model=model.state_dict(), config=cfg, provenance=provenance,
                            optimizer_steps=steps), output / name)
    return dict(output=str(output), checkpoint=str(output / "last.pt"), steps=steps)


def probe_native(config_path, steps=2, batch_size=1):
    if steps < 2 or batch_size < 1:
        raise ValueError("Probe needs >=2 steps (zero-residual branches open after step 1)")
    cfg = load_native_config(config_path)
    seed_all(cfg.get("seed", 42))
    records = read_manifest(cfg["train_manifest"])
    if len(records) < batch_size:
        raise ValueError("Probe batch exceeds available training images")
    dataset = PolypUNetDataset(records[:batch_size], augment=False)
    images, target = zip(*(dataset[index] for index in range(len(dataset))))
    device = torch.device(cfg.get("device", "cuda"))
    model = build_native_model(cfg, device).train()
    optimizer = _optimizer(model, cfg)
    images, target = torch.stack(images).to(device), torch.stack(target).to(device)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(model, images, target, cfg)
        loss.backward()
        history.append(float(loss.detach()))
        optimizer.step()
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Nonfinite gradient in {name}")
            group = ("adapter" if "prompt_learn" in name else
                     "scds" if name.startswith("scds.") else
                     "sbca" if cfg["modules"]["sbca"]
                     and "layers.0.cross_attn_token_to_image." in name and ".base." not in name
                     else "rcsr" if name.startswith("rcsr.") else "native")
            groups[group] = groups.get(group, 0.) + parameter.grad.detach().float().square().sum().item()
    return dict(losses=history, gradient_norms={key: value**.5 for key, value in groups.items()},
                output_shape=[batch_size, 1, 352, 352], modules=cfg["modules"],
                peak_cuda_gb=torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda" else None)


@torch.inference_mode()
def predict_native(checkpoint_path, manifest, output, device="cuda", normalization="raw"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if normalization not in {"raw", "per-image"}:
        raise ValueError("Unknown probability normalization")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("architecture") != "native_sam2_hiera_l":
        raise ValueError("Expected project native-SAM2 checkpoint, not SAM2-UNet")
    model = build_native_model(state["config"], device, allow_random=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    records = read_manifest(manifest, require_masks=False)
    output.mkdir(parents=True)
    mapping = []
    for index, record in enumerate(records):
        with Image.open(record.image) as im:
            image = im.convert("RGB")
        x = torch.from_numpy(np.array(image.resize((352, 352), Image.Resampling.BILINEAR),
                                      dtype=np.float32) / 255).permute(2, 0, 1)
        x = (x - x.new_tensor([.485, .456, .406])[:, None, None]) / x.new_tensor(
            [.229, .224, .225])[:, None, None]
        logits = model(x[None].to(device))
        prob = F.interpolate(logits, (image.height, image.width), mode="bilinear",
                             align_corners=False).sigmoid()[0, 0].cpu().numpy()
        if normalization == "per-image":
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        name = f"{index:06d}.png"
        Image.fromarray((prob * 255).astype(np.uint8)).save(output / name)
        mapping.append(dict(id=record.sample_id, image=str(record.image), prediction=name))
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "prediction"])
        writer.writeheader()
        writer.writerows(mapping)
    (output / "provenance.json").write_text(json.dumps(dict(
        checkpoint=str(Path(checkpoint_path).resolve()), normalization=normalization,
        encoding="uint8_floor", native_grid=True, modules=state["config"]["modules"],
        count=len(records), epoch=state["epoch"],
    ), indent=2))
    return dict(output=str(output), count=len(records))
