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
