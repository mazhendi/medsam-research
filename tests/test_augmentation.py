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


@pytest.mark.parametrize('angle', [0., 31., -47., 90.])
@pytest.mark.parametrize('flips', [(.1, .1), (.1, .9), (.9, .1), (.9, .9)])
def test_hsnet_shared_geometry_and_no_color_jitter(tmp_path, monkeypatch, angle, flips):
    records = make_record(tmp_path)
    dataset = BinaryDataset(records, 64, augmentation='hsnet')
    # Identical pixel landmarks in image/mask/FOV must stay identical after the
    # continuous-angle rotation and both flips, including rectangular images.
    landmarks = np.zeros((17, 29), dtype=np.uint8)
    landmarks[2:8, 3:10] = 1
    landmarks[12:15, 21:26] = 1
    rgb = Image.fromarray(np.repeat((landmarks*255)[:, :, None], 3, axis=2))
    monkeypatch.setattr(random, 'uniform', lambda a, b: angle)
    draws = iter(flips)
    monkeypatch.setattr(random, 'random', lambda: next(draws))
    image, mask, valid = dataset._augment_hsnet(rgb, landmarks, landmarks)
    np.testing.assert_array_equal(np.asarray(image)[:, :, 0], mask*255)
    np.testing.assert_array_equal(mask, valid)
    assert mask.shape == landmarks.shape
    assert set(np.unique(image)) <= {0, 255}  # No brightness/contrast changes.


def test_hsnet_loader_train_only_and_corner_policy(tmp_path, monkeypatch):
    from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig
    from medsam_research.engine import make_loader

    records = make_record(tmp_path)
    cfg = Config(ModelConfig(), DataConfig(), TrainingConfig(
        augmentation='hsnet', num_workers=0, device='cpu', batch_size=1))
    train = make_loader(records, 32, cfg, augment=True)
    val = make_loader(records, 32, cfg)
    assert train.dataset.augmentation == 'hsnet'
    assert val.dataset.augmentation == 'none'
    monkeypatch.setattr(random, 'uniform', lambda a, b: 45.)
    monkeypatch.setattr(random, 'random', lambda: 1.)
    # Without an explicit FOV, fill corners remain supervised background.
    sample = train.dataset[0]
    assert sample['native_valid'].all()
    # With an explicit FOV, pixels rotated out of the source stay invalid.
    fov = tmp_path/'fov.png'
    Image.fromarray(np.full((24, 32), 255, dtype=np.uint8)).save(fov)
    from dataclasses import replace
    with_fov = BinaryDataset([replace(records[0], fov=fov)], 32, augmentation='hsnet')[0]
    assert with_fov['native_valid'][0, 0] == 0
    assert set(with_fov['mask'].unique().tolist()) <= {0., 1.}


def test_hsnet_config_is_independent_from_legacy_light():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    current = load_config(root/'configs/polyp_full_hsnet.yaml')
    previous = load_config(root/'configs/polyp_full_aug.yaml')
    assert current.training.augmentation == 'hsnet'
    assert previous.training.augmentation == 'light'
    assert current.training.output_dir != previous.training.output_dir
