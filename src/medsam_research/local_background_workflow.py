"""Compute-matched training controls for local positives and background selection."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import load_pair, read_manifest
from .local_background import LocalBackgroundDataset, local_training_step
from .native_sam2 import build_native_model
from .native_sam2_workflow import _optimizer, load_native_config, seed_all
from .sam2_unet_workflow import seed_worker


def load_local_config(path):
    cfg = load_native_config(path)
    local = cfg.get("local_background", {})
    allowed = {"mode", "candidates", "attempts", "margin_fraction", "warmup_epochs"}
    if set(local) != allowed or local["mode"] not in {"random", "hard", "matched_full"}:
        raise ValueError("Specify all local_background options and random/hard/matched_full mode")
    if any(cfg["modules"].values()) or cfg.get("shrink_augmentation", {}).get("probability", 0):
        raise ValueError("Local-background controls require modules OFF and shrink OFF")
    if (type(local["candidates"]) is not int or local["candidates"] < 1
            or type(local["attempts"]) is not int or local["attempts"] < 1
            or not 0 <= local["margin_fraction"] <= .1
            or type(local["warmup_epochs"]) is not int or local["warmup_epochs"] < 0):
        raise ValueError("Invalid local-background sampler configuration")
    return cfg


def _dataset(records, cfg):
    local = cfg["local_background"]
    return LocalBackgroundDataset(records, seed=cfg.get("seed", 42),
                                  candidates=local["candidates"], attempts=local["attempts"],
                                  margin_fraction=local["margin_fraction"])


def train_local(config_path):
    cfg = load_local_config(config_path)
    output = Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    seed_all(cfg.get("seed", 42))
    records = read_manifest(cfg["train_manifest"])
    if len(records) != 1450:
        raise ValueError(f"Expected 1450 training images; got {len(records)}")
    for record in records:
        _, mask = load_pair(record)
        if not mask.any():
            raise ValueError(f"Original pool needs nonempty GT: {record.sample_id}")
    model = build_native_model(cfg, cfg.get("device", "cuda"))
    optimizer = _optimizer(model, cfg)
    epochs = cfg.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-7)
    dataset = _dataset(records, cfg)
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 16), shuffle=True,
                        num_workers=cfg.get("workers", 4), worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
                        pin_memory=str(cfg.get("device", "cuda")).startswith("cuda"))
    output.mkdir(parents=True)
    provenance = dict(config=cfg, train_count=1450, selection="fixed_final_epoch", precision="fp32",
                      manifest_sha256=hashlib.sha256(Path(cfg["train_manifest"]).read_bytes()).hexdigest(),
                      gradient_forwards_per_step=3,
                      scoring_forwards_per_step=cfg["local_background"]["candidates"],
                      background_loss="BCE; full-image fallback uses structure loss",
                      score="mean sigmoid probability on resized candidate crop",
                      prediction="unchanged full-image native SAM2 inference")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    steps = 0
    for epoch in range(1, epochs + 1):
        dataset.epoch = epoch
        lr = optimizer.param_groups[0]["lr"]
        totals = {}
        for batch in loader:
            stats = local_training_step(model, batch, optimizer, cfg["local_background"]["mode"],
                                        epoch > cfg["local_background"]["warmup_epochs"])
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.) + value
            steps += 1
        scheduler.step()
        row = dict(epoch=epoch, optimizer_steps=steps, learning_rate=lr,
                   gradient_example_exposures=int(3 * totals["examples"]),
                   background_fallbacks=int(totals["background_fallbacks"]),
                   background_count=int(totals["background_count"]),
                   selected_background_score=totals["background_score_sum"]
                   / max(1, totals["background_count"]),
                   forward_calls=int(totals["forward_calls"]),
                   backward_calls=int(totals["backward_calls"]))
        row.update({key: totals[key] / len(loader) for key in
                    ("loss", "full_loss", "positive_loss", "background_loss")})
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch % cfg.get("save_every", 5) == 0 or epoch == epochs:
            name = "last.pt" if epoch == epochs else f"epoch_{epoch:03d}.pt"
            torch.save(dict(architecture=cfg["architecture"], epoch=epoch, model=model.state_dict(),
                            config=cfg, provenance=provenance, optimizer_steps=steps), output / name)
    return dict(checkpoint=str(output / "last.pt"), steps=steps)


def probe_local(config_path, steps=2, batch_size=1):
    if steps < 1 or batch_size < 1:
        raise ValueError("Probe steps and batch size must be positive")
    cfg = load_local_config(config_path)
    seed_all(cfg.get("seed", 42))
    records = read_manifest(cfg["train_manifest"])
    if len(records) < batch_size:
        raise ValueError("Not enough probe records")
    loader = DataLoader(_dataset(records[:batch_size], cfg), batch_size=batch_size, num_workers=0)
    batch = next(iter(loader))
    model = build_native_model(cfg, cfg.get("device", "cuda"))
    optimizer = _optimizer(model, cfg)
    history = [local_training_step(model, batch, optimizer, cfg["local_background"]["mode"],
                                    use_hard=True) for _ in range(steps)]
    return dict(mode=cfg["local_background"]["mode"], steps=history,
                peak_cuda_gb=torch.cuda.max_memory_allocated() / 1024**3
                if str(cfg.get("device", "cuda")).startswith("cuda") else None)
