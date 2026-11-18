import csv
import random

import numpy as np
import pytest
import yaml
from PIL import Image

from medsam_research.config import load_config
from medsam_research.data import BinaryDataset, read_manifest


def make_record(tmp_path):
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:12, :16] = 120
    image[12:, 16:] = 200
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:10, 6:14] = 255
    Image.fromarray(image).save(tmp_path / "a.png")
    Image.fromarray(mask).save(tmp_path / "a_mask.png")
    row = {"id": "a", "image": "a.png", "mask": "a_mask.png",
           "dataset": "synthetic", "group": ""}
    path = tmp_path / "train.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return read_manifest(path)


def test_light_augmentation_changes_image_and_mask_together(tmp_path):
    records = make_record(tmp_path)
    plain = BinaryDataset(records, 64)
    augmented = BinaryDataset(records, 64, augmentation="light")
    random.seed(0)
    first = augmented[0]
    second = augmented[0]
    assert not first["image"].equal(second["image"]) or not first["mask"].equal(second["mask"])
    assert first["mask"].shape == (1, 64, 64)
    assert set(first["mask"].unique().tolist()) <= {0.0, 1.0}
    assert first["native_mask"].shape == first["native_valid"].shape
    # 90-degree rotation keeps the native footprint consistent for both arrays.
    assert first["native_mask"].shape[0] in (24, 32)
    reference = plain[0]
    assert reference["mask"].shape == first["mask"].shape


def test_augmentation_none_is_deterministic(tmp_path):
    records = make_record(tmp_path)
    dataset = BinaryDataset(records, 64)
    assert dataset[0]["image"].equal(dataset[0]["image"])
    assert dataset[0]["mask"].equal(dataset[0]["mask"])


def test_invalid_augmentation_rejected(tmp_path):
    records = make_record(tmp_path)
    with pytest.raises(ValueError):
        BinaryDataset(records, 64, augmentation="heavy")


def test_config_accepts_and_rejects_augmentation(tmp_path):
    records = make_record(tmp_path)
    base = {
        "model": {"checkpoint": str(tmp_path / "ckpt.pt")},
        "data": {"train_manifest": str(tmp_path / "train.csv"),
                 "val_manifest": str(tmp_path / "train.csv")},
        "training": {"output_dir": "out", "augmentation": "light"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base))
    assert load_config(path).training.augmentation == "light"
    base["training"]["augmentation"] = "heavy"
    path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError):
        load_config(path)
    assert records  # keep fixture usage explicit
