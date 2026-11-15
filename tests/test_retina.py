import csv
import gzip
import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from medsam_research.data import BinaryDataset, audit_manifests, read_manifest
from medsam_research.metrics import binary_metrics, segmentation_loss
from medsam_research.retina import CHASE_SHA256, STARE_IDS, prepare_retina


def encoded(array, image_format):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format=image_format)
    return stream.getvalue()


def test_bundled_chase_real_data_group_split_and_observer(tmp_path):
    output = tmp_path / "chase"
    report = prepare_retina("CHASE_DB1", None, output)
    assert report["sample_count"] == 28
    assert report["sources"][0]["sha256"] == CHASE_SHA256
    records = {name: read_manifest(output / f"{name}.csv") for name in ("train", "val", "test")}
    assert [len(records[s]) for s in records] == [16, 4, 8]
    groups = [set(r.group for r in rows) for rows in records.values()]
    assert [len(g) for g in groups] == [8, 2, 4]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    with Image.open(output / "images/Image_01L.png") as image:
        assert image.size == (999, 960)
    archive = Path(__file__).parents[1] / "assets/datasets/CHASE_DB1/CHASEDB1.zip"
    with zipfile.ZipFile(archive) as zipped:
        with Image.open(io.BytesIO(zipped.read("Image_01L_1stHO.png"))) as im:
            expected = np.asarray(im.convert("L")) > 0
        with Image.open(io.BytesIO(zipped.read("Image_01L.jpg"))) as im:
            original = np.asarray(im.convert("RGB"))
    with Image.open(output / "masks1/Image_01L.png") as im:
        np.testing.assert_array_equal(np.asarray(im) > 0, expected)
    with Image.open(output / "images/Image_01L.png") as im:
        np.testing.assert_array_equal(np.asarray(im), original)
    assert len(list((output / "masks2").glob("*.png"))) == 28
    assert all(r.fov is None for rows in records.values() for r in rows)
    with pytest.raises(FileExistsError):
        prepare_retina("CHASE_DB1", None, output)


@pytest.mark.parametrize("test_labels", [True, False])
def test_drive_preserves_official_test_and_fov(tmp_path, test_labels):
    source = tmp_path / "raw"
    source.mkdir()
    for index in range(1, 41):
        split = "test" if index <= 20 else "training"
        image = np.random.default_rng(index).integers(0, 256, (17, 19, 3), dtype=np.uint8)
        Image.fromarray(image).save(source / f"{index:02d}_{split}.tif")
        mask = np.zeros((17, 19), np.uint8)
        mask[4:10, 5:11] = 255
        if split == "training" or test_labels:
            Image.fromarray(mask).save(source / f"{index:02d}_manual1.gif")
        fov = np.zeros_like(mask)
        fov[2:-2, 2:-2] = 255
        Image.fromarray(fov).save(source / f"{index:02d}_{split}_mask.gif")
    output = tmp_path / "drive"
    report = prepare_retina("DRIVE", [source], output)
    assert len(report["splits"]["train"]) == 16
    assert len(report["splits"]["val"]) == 4
    assert all(int(i) >= 21 for s in ("train", "val") for i in report["splits"][s])
    test_name = "test" if test_labels else "test_inference"
    assert set(report["splits"][test_name]) == {f"{i:02d}" for i in range(1, 21)}
    dataset = BinaryDataset(read_manifest(output / "train.csv"), 32)
    item = dataset[0]
    assert item["fov_used"] and item["valid"].sum() < 32 * 32
    assert item["native_valid"].sum() == 13 * 15
    if not test_labels:
        with (output / "test_inference.csv").open() as stream:
            assert all(not r["mask"] for r in csv.DictReader(stream))
        assert len(read_manifest(output / "test_inference.csv", require_masks=False)) == 20


def test_stare_nested_ppm_gzip_and_separate_observers(tmp_path):
    sources = []
    for who in ("images", "ah", "vk"):
        path = tmp_path / f"{who}.tar"
        with tarfile.open(path, "w") as archive:
            for index, sid in enumerate(sorted(STARE_IDS)):
                if who == "images":
                    array = np.random.default_rng(index).integers(
                        0, 256, (15, 21, 3), dtype=np.uint8
                    )
                    name = f"nested/{sid}.ppm.gz"
                else:
                    array = np.zeros((15, 21, 3), dtype=np.uint8)
                    array[3:9, 5 : 12 if who == "ah" else 14] = 255
                    name = f"labels/{sid}.{who}.ppm.gz"
                raw = gzip.compress(encoded(array, "PPM"))
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        sources.append(path)
    output = tmp_path / "stare"
    report = prepare_retina("STARE", sources, output, observer=2)
    assert set(report["splits"]) == {"external"}
    rows = read_manifest(output / "external.csv")
    assert len(rows) == 20 and all(r.mask.parent.name == "masks2" for r in rows)
    assert all(r.group == "" and r.fov is None for r in rows)
    assert not (output / "train.csv").exists()
    assert audit_manifests([output / "external.csv"])[0]["samples_without_group"] == 20


def test_archive_traversal_is_rejected_without_writes(tmp_path):
    source = tmp_path / "malicious.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../outside.png", b"not an image")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        prepare_retina("CHASE_DB1", [source], tmp_path / "out")
    assert not (tmp_path / "outside.png").exists()
    assert not (tmp_path / "out").exists()


def test_fov_excludes_errors_and_training_gradients():
    target = np.zeros((10, 10), dtype=bool)
    target[3:7, 3:7] = True
    prediction = target.copy()
    prediction[:2] = True
    valid = np.zeros_like(target)
    valid[2:8, 2:8] = True
    assert binary_metrics(prediction, target)["dice"] < 1
    assert binary_metrics(prediction, target, valid=valid)["dice"] == 1
    logits = torch.randn(1, 1, 10, 10, requires_grad=True)
    weights = torch.from_numpy(valid)[None, None]
    loss = segmentation_loss(logits, torch.from_numpy(target)[None, None], weights)
    loss.backward()
    assert logits.grad[~weights].abs().sum() == 0
    assert logits.grad[weights].abs().sum() > 0


def test_missing_dataset_is_not_silently_downloaded(tmp_path):
    with pytest.raises(ValueError, match="Supply downloaded"):
        prepare_retina("DRIVE", None, tmp_path / "out")
