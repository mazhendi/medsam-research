import csv
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from medsam_research.data import read_manifest
from medsam_research.native_sam2_workflow import load_native_config
from medsam_research.sam2_unet_workflow import PolypUNetDataset, shrink_pair


def make_records(tmp_path):
    image = np.zeros((32, 32, 3), np.uint8)
    mask = np.zeros((32, 32), np.uint8)
    image[8:24, 8:24] = 255
    mask[8:24, 8:24] = 255
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    path = tmp_path / "train.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "mask"])
        writer.writeheader()
        writer.writerow(dict(id="case", image="image.png", mask="mask.png"))
    return read_manifest(path)


def test_shrink_keeps_alignment_and_never_copies_lesion_into_padding(tmp_path):
    records = make_records(tmp_path)
    image, mask = Image.open(records[0].image), Image.open(records[0].mask)
    x, y = shrink_pair(image, mask, .5, 3, 7)
    a, b = np.asarray(x), np.asarray(y)
    assert x.size == image.size and y.size == mask.size
    assert np.count_nonzero(b >= 128) == 64
    assert a[12, 8, 0] > 200 and b[12, 8] > 200
    assert (b[:7] == 0).all() and (b[:, :3] == 0).all()
    assert (a[:7] == 64).all()  # mean RGB padding, no reflected duplicate foreground


def test_disabled_and_inference_paths_preserve_legacy_rng_and_data(tmp_path):
    records = make_records(tmp_path)
    random.seed(42)
    baseline = PolypUNetDataset(records, size=32)[0]
    after = random.getstate()
    random.seed(42)
    disabled = PolypUNetDataset(records, size=32, shrink_probability=0)[0]
    assert random.getstate() == after
    assert all(torch.equal(a, b) for a, b in zip(baseline, disabled))
    plain = PolypUNetDataset(records, size=32, augment=False)[0]
    inference = PolypUNetDataset(records, size=32, augment=False, shrink_probability=1)[0]
    assert all(torch.equal(a, b) for a, b in zip(plain, inference))


def test_shrink_uses_independent_rng_and_preserves_positive_target(tmp_path):
    records = make_records(tmp_path)
    random.seed(42)
    PolypUNetDataset(records, size=32)[0]
    expected_state = random.getstate()
    random.seed(42)
    _, target = PolypUNetDataset(records, size=32, shrink_probability=1,
                                 shrink_min=.5, shrink_max=.5)[0]
    assert random.getstate() == expected_state
    assert (target >= .5).sum() == 64
    with pytest.raises(ValueError):
        PolypUNetDataset(records, shrink_probability=1.2)


def test_shrink_config_changes_only_augmentation_and_output():
    root = Path(__file__).resolve().parents[1] / "configs"
    baseline = load_native_config(root / "native_sam2_baseline.yaml")
    control = load_native_config(root / "native_sam2_shrink.yaml")
    assert control.pop("shrink_augmentation")["probability"] == .3
    assert baseline.pop("output") != control.pop("output")
    assert control == baseline
