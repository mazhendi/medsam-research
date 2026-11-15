import csv

import numpy as np
import pytest
import torch
from PIL import Image

from medsam_research.config import load_config
from medsam_research.data import BinaryDataset, audit_manifests, read_manifest
from medsam_research.metrics import binary_metrics, native_prediction, segmentation_loss


def make_manifest(tmp_path, name, ids, groups=None):
    rows = []
    for index, sid in enumerate(ids):
        # Different image content across manifests, independent of image filenames.
        value = sum(sid.encode()) % 250
        image = np.full((13 + index, 19, 3), value, dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[3:7, 4:9] = 255
        Image.fromarray(image).save(tmp_path / f"{sid}.png")
        Image.fromarray(mask).save(tmp_path / f"{sid}_mask.png")
        rows.append(
            {
                "id": sid,
                "image": f"{sid}.png",
                "mask": f"{sid}_mask.png",
                "dataset": "synthetic",
                "group": groups[index] if groups else "",
            }
        )
    path = tmp_path / name
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_native_masks_and_binary_resize(tmp_path):
    path = make_manifest(tmp_path, "train.csv", ["a"])
    item = BinaryDataset(read_manifest(path), 32)[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["native_mask"].shape == (13, 19)
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}


def test_duplicate_pixels_across_formats_are_rejected(tmp_path):
    train = make_manifest(tmp_path, "train.csv", ["a"])
    val = make_manifest(tmp_path, "val.csv", ["b"])
    with Image.open(tmp_path / "a.png") as im:
        im.save(tmp_path / "b.png")
    with pytest.raises(ValueError, match="Duplicate image"):
        audit_manifests([train, val])


def test_patient_overlap_rejected(tmp_path):
    train = make_manifest(tmp_path, "train.csv", ["a"], ["patient1"])
    val = make_manifest(tmp_path, "val.csv", ["b"], ["patient1"])
    with pytest.raises(ValueError, match="group occurs across splits"):
        audit_manifests([train, val])


def test_multiclass_mask_rejected(tmp_path):
    path = make_manifest(tmp_path, "train.csv", ["a"])
    mask = np.zeros((13, 19), dtype=np.uint8)
    mask[0, 0], mask[0, 1] = 1, 2
    Image.fromarray(mask).save(tmp_path / "a_mask.png")
    with pytest.raises(ValueError, match="not a binary mask"):
        audit_manifests([path])


def test_metrics_empty_and_wrong_and_perfect():
    empty = np.zeros((10, 12), dtype=bool)
    full = ~empty
    assert binary_metrics(empty, empty)["dice"] == 1
    assert binary_metrics(empty, full)["boundary_f1"] == 0
    assert binary_metrics(full, full)["boundary_f1"] == 1
    assert binary_metrics(full, full)["iou"] == 1


def test_boundary_distance_and_dice_are_distinct():
    a = np.zeros((20, 20), dtype=bool)
    a[4:12, 4:12] = True
    b = np.roll(a, 1, axis=1)
    scores = binary_metrics(b, a, tolerance=1)
    assert scores["dice"] < 1
    assert scores["boundary_f1"] == 1
    assert binary_metrics(b, a, tolerance=0)["boundary_f1"] < 1


def test_resize_before_threshold_and_loss_backprop():
    logits = torch.tensor([[[[-8.0, 8.0], [-8.0, 8.0]]]], requires_grad=True)
    mask = native_prediction(logits.detach(), (9, 13), 0.5)
    assert mask.shape == (9, 13)
    assert not mask[:, 0].any() and mask[:, -1].all()
    loss = segmentation_loss(logits, torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_config_paths_and_unknown_options(tmp_path):
    import yaml

    raw = {
        "model": {"checkpoint": "weights.pt"},
        "data": {"train_manifest": "train.csv", "val_manifest": "val.csv"},
        "training": {"output_dir": "results"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    assert cfg.model.checkpoint == str(tmp_path / "weights.pt")
    raw["training"]["action_selector"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_manifest_suffix_pairing_and_no_overwrite(tmp_path):
    from medsam_research.manifest import create_manifest

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((12, 15, 3), dtype=np.uint8)).save(images / "case.jpg")
    Image.fromarray(np.zeros((12, 15), dtype=np.uint8)).save(masks / "case_segmentation.png")
    out = tmp_path / "lists" / "all.csv"
    assert create_manifest(images, masks, out, "skin", "_segmentation") == 1
    assert read_manifest(out)[0].image == images / "case.jpg"
    assert audit_manifests([out])[0]["empty_masks"] == 1
    with pytest.raises(FileExistsError):
        create_manifest(images, masks, out, "skin", "_segmentation")


def test_high_bit_depth_image_requires_explicit_conversion(tmp_path):
    path = make_manifest(tmp_path, "train.csv", ["a"])
    Image.fromarray(np.full((13, 19), 12000, dtype=np.uint16)).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="8-bit"):
        audit_manifests([path])
