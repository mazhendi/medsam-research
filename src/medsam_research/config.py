"""Small, strict configuration schema. No unimplemented options are silently accepted."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


@dataclass
class ModelConfig:
    sam_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    checkpoint: str = ""
    prompt_tokens: int = 4


@dataclass
class DataConfig:
    train_manifest: str = ""
    val_manifest: str = ""
    mask_threshold: int = 0


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 2
    amp: bool = True
    device: str = "cuda"
    threshold: float = 0.5
    boundary_tolerance: int = 2
    output_dir: str = "outputs/run"


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    def to_dict(self):
        return asdict(self)


def _construct(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"model", "data", "training"}:
        raise ValueError("Config must contain exactly model, data, training")
    cfg = Config(
        _construct(ModelConfig, raw["model"]),
        _construct(DataConfig, raw["data"]),
        _construct(TrainingConfig, raw["training"]),
    )
    for obj, key in [
        (cfg.model, "checkpoint"),
        (cfg.data, "train_manifest"),
        (cfg.data, "val_manifest"),
        (cfg.training, "output_dir"),
    ]:
        value = getattr(obj, key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty path")
        setattr(obj, key, str((path.parent / value).resolve()))
    t = cfg.training
    if cfg.model.prompt_tokens < 1 or t.epochs < 1 or t.batch_size < 1:
        raise ValueError("prompt_tokens, epochs, batch_size must be positive")
    if t.learning_rate <= 0 or t.weight_decay < 0 or t.num_workers < 0:
        raise ValueError("Invalid learning rate, weight decay, or worker count")
    if not 0 < t.threshold < 1 or t.boundary_tolerance < 0:
        raise ValueError("threshold must be in (0,1); boundary_tolerance must be >= 0")
    if not 0 <= cfg.data.mask_threshold <= 255:
        raise ValueError("mask_threshold must be in [0,255]")
    return cfg
