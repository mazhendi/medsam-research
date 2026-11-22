import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from medsam_research import polyp_pvt_workflow as workflow
from medsam_research.data import Record
from medsam_research.polyp_pvt import build_polyp_pvt


def write_config(tmp_path):
    cfg = dict(architecture="polyp_pvt_b2", upstream="upstream", encoder_checkpoint="encoder.pth",
               train_manifest="train.csv", output="run", expected_train_count=2, image_size=352,
               epochs=1, batch_size=2, learning_rate=.001, weight_decay=.0001,
               gradient_clip=.5, augmentation="none", scales=[.75, 1., 1.25],
               workers=0, seed=42, device="cpu")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def write_data(tmp_path):
    records = []
    for i in range(2):
        image, mask = tmp_path / f"image{i}.png", tmp_path / f"mask{i}.png"
        rgb = np.zeros((11, 17, 3), dtype=np.uint8)
        rgb[:, :, 0] = 50 + i * 30
        target = np.zeros((11, 17), dtype=np.uint8)
        target[3:8, 4:10] = 255
        Image.fromarray(rgb).save(image)
        Image.fromarray(target).save(mask)
        records.append(Record(str(i), image, mask, "test", str(i)))
    with (tmp_path / "train.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image", "mask", "dataset", "group"])
        for r in records:
            writer.writerow([r.sample_id, r.image, r.mask, r.dataset, r.group])
    return records


class TinyTwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(3, 1, 1)
        self.second = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.first(x), self.second(x)


def test_soft_mask_resize_and_normalization(tmp_path):
    records = write_data(tmp_path)
    x, y = workflow.PolypPVTDataset(records, 32)[0]
    assert x.shape == (3, 32, 32) and y.shape == (1, 32, 32)
    assert ((y > 0) & (y < 1)).any()
    assert torch.isclose(x[0, 0, 0], torch.tensor((50 / 255 - .485) / .229))


def test_three_updates_train_both_heads_without_test_split(tmp_path, monkeypatch):
    path, _ = write_config(tmp_path)
    write_data(tmp_path)
    (tmp_path / "encoder.pth").write_bytes(b"probe")
    model = TinyTwoHead()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(workflow, "build_polyp_pvt", lambda *args: model)
    report = workflow.train_pvt(path)
    assert report["steps"] == 3
    for name, value in model.state_dict().items():
        assert not torch.equal(value, before[name])
    with pytest.raises(FileExistsError):
        workflow.train_pvt(path)
    saved = torch.load(report["checkpoint"], weights_only=True)
    assert saved["provenance"]["selection"] == "fixed_final_epoch_no_test_access"


def test_prediction_sums_logits_and_preserves_native_shape(tmp_path, monkeypatch):
    path, _ = write_config(tmp_path)
    write_data(tmp_path)
    model = TinyTwoHead()
    for head in (model.first, model.second):
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, 1.)
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(architecture="polyp_pvt_b2", model=model.state_dict(), epoch=1), checkpoint)
    monkeypatch.setattr(workflow, "build_polyp_pvt", lambda *args: TinyTwoHead())
    output = tmp_path / "predictions"
    report = workflow.predict_pvt(path, checkpoint, tmp_path / "train.csv", output)
    assert report["count"] == 2
    with Image.open(output / "000000.png") as image:
        pixels = np.asarray(image)
    assert pixels.shape == (11, 17)
    assert pixels.min() == pixels.max() == int(torch.sigmoid(torch.tensor(2.)).item() * 255)
    with pytest.raises(ValueError, match="normalization"):
        workflow.predict_pvt(path, checkpoint, tmp_path / "train.csv", tmp_path / "bad", "other")


def test_source_hash_refuses_modified_algorithm(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/pvt.py").write_text("modified")
    with pytest.raises(ValueError, match="differs"):
        build_polyp_pvt(tmp_path)


def test_invalid_training_config(tmp_path):
    path, cfg = write_config(tmp_path)
    cfg["augmentation"] = "light"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="augmentation"):
        workflow.load_pvt_config(path)


def test_default_full_and_short_budgets():
    root = Path(__file__).resolve().parents[1]
    full = workflow.load_pvt_config(root / "configs/polyp_pvt_1450.yaml")
    short = workflow.load_pvt_config(root / "configs/polyp_pvt_20ep.yaml")
    assert full["epochs"] == 99 and short["epochs"] == 20
    assert full["scales"] == short["scales"] == [.75, 1., 1.25]
    assert full["output"] != short["output"]


@pytest.mark.parametrize("size", [352, 448, 512])
def test_resolution_override_uses_requested_size_and_native_output(tmp_path, monkeypatch, size):
    import json

    path, _ = write_config(tmp_path)
    write_data(tmp_path)
    model = TinyTwoHead()
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(architecture="polyp_pvt_b2", model=model.state_dict(), epoch=1,
                    provenance={"config": {"image_size": 352}}), checkpoint)
    shapes = []
    model.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    monkeypatch.setattr(workflow, "build_polyp_pvt", lambda *args: model)
    output = tmp_path / "predictions"
    workflow.predict_pvt(path, checkpoint, tmp_path / "train.csv", output, inference_size=size)
    assert shapes == [(1, 3, size, size)] * 2
    with Image.open(output / "000000.png") as image:
        assert image.size == (17, 11)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["inference_size"] == size
    assert provenance["training_image_size"] == 352
    assert provenance["elapsed_seconds"] >= 0


@pytest.mark.parametrize("size", [0, 33, 351, 353, -32, True, 352.0])
def test_invalid_resolution_rejected_before_file_access(size):
    with pytest.raises(ValueError, match="inference_size"):
        workflow.predict_pvt("absent", "absent", "absent", "absent", inference_size=size)


def test_sweep_keeps_five_sets_and_three_resolutions_independent(tmp_path, monkeypatch):
    path, _ = write_config(tmp_path)
    write_data(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    for name in workflow.POLYP_TEST_SETS:
        (tmp_path / f"{name}_dedup.csv").write_bytes((tmp_path / "train.csv").read_bytes())
    calls = []

    def fake_predict(config, state, manifest, output, normalization, inference_size):
        calls.append((state, manifest.name, inference_size, normalization, output))
        output.mkdir(parents=True)
        return dict(output=str(output), count=2, inference_size=inference_size)

    monkeypatch.setattr(workflow, "predict_pvt", fake_predict)
    report = workflow.sweep_pvt(path, checkpoint, tmp_path, tmp_path / "sweep")
    assert len(calls) == len(report["results"]) == 15
    assert {call[0] for call in calls} == {checkpoint.resolve()}
    assert {call[2] for call in calls} == {352, 448, 512}
    assert len({call[4] for call in calls}) == 15
    assert report["selection"] == "none_diagnostic_only"
    assert (tmp_path / "sweep/sweep.json").is_file()


def test_sweep_missing_set_fails_before_output(tmp_path):
    path, _ = write_config(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "sweep"
    with pytest.raises(FileNotFoundError):
        workflow.sweep_pvt(path, checkpoint, tmp_path, output)
    assert not output.exists()
