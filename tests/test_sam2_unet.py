import csv

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from medsam_research.data import read_manifest
from medsam_research.sam2_unet import SAM2UNet
from medsam_research.sam2_unet_workflow import (
    PolypUNetDataset,
    merge_training_manifests,
    predict_unet,
    scaled_size,
    structure_loss,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24)

    def forward(self, x):
        return x + self.attn.qkv(x)[..., :8]


class TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.blocks = nn.Sequential(Block())

    def forward(self, x):
        x = self.blocks(self.stem(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return [torch.nn.functional.adaptive_avg_pool2d(x, x.shape[-1] // scale)
                .repeat(1, channels // 8, 1, 1)
                for scale, channels in zip((4, 8, 16, 32), (144, 288, 576, 1152))]


def test_deep_supervision_reaches_adapter_through_frozen_block():
    torch.set_num_threads(2)
    model = SAM2UNet(encoder=TinyTrunk())
    x = torch.randn(2, 3, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > .6).float()
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()
              if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    outputs = model(x)
    assert [p.shape for p in outputs] == [target.shape] * 3
    loss = sum(structure_loss(p, target) for p in outputs)
    loss.backward()
    assert model.encoder.blocks[0].prompt_learn[0].weight.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0
    assert model.side1.weight.grad.abs().sum() > 0
    assert model.side2.weight.grad.abs().sum() > 0
    optimizer.step()
    for name, before in frozen.items():
        assert torch.equal(dict(model.named_parameters())[name], before)
        assert dict(model.named_parameters())[name].grad is None


def make_manifest(root, name, value):
    image = root / f"{name}.png"
    mask = root / f"{name}_mask.png"
    Image.fromarray(np.full((15, 17, 3), value, np.uint8)).save(image)
    y = np.zeros((15, 17), np.uint8)
    y[4:9, 5:10] = 255
    Image.fromarray(y).save(mask)
    path = root / f"{name}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "mask"])
        writer.writeheader()
        writer.writerow(dict(id=name, image=image.name, mask=mask.name))
    return path


def test_merge_rejects_wrong_count_and_duplicate_pixels(tmp_path):
    a = make_manifest(tmp_path, "a", 10)
    b = make_manifest(tmp_path, "b", 10)
    with pytest.raises(ValueError, match="1450"):
        merge_training_manifests([a, b], tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="Duplicate"):
        merge_training_manifests([a, b], tmp_path / "bad.csv", expected_count=2)
    assert not (tmp_path / "bad.csv").exists()


def test_merge_preserves_paths_and_author_soft_targets(tmp_path):
    a = make_manifest(tmp_path, "a", 10)
    b = make_manifest(tmp_path, "b", 20)
    out = tmp_path / "nested" / "all.csv"
    assert merge_training_manifests([a, b], out, expected_count=2)["count"] == 2
    records = read_manifest(out)
    x, y = PolypUNetDataset(records, size=32, augment=False)[0]
    assert x.shape == (3, 32, 32)
    assert y.shape == (1, 32, 32)
    assert ((y > 0) & (y < 1)).any()
    with pytest.raises(FileExistsError):
        merge_training_manifests([a, b], out, expected_count=2)


def test_scale_and_loss_contract():
    assert scaled_size(352, 1) == 352
    assert scaled_size(352, 1.25) == 448
    y = torch.zeros(1, 1, 32, 32)
    y[:, :, 10:20, 10:20] = 1
    good = (2 * y - 1) * 8
    assert structure_loss(good, y) < structure_loss(-good, y)


def test_prediction_exports_native_grid_mapping_without_mask(tmp_path, monkeypatch):
    import medsam_research.sam2_unet as architecture

    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(3, 1, 1)

        def forward(self, x):
            return (self.head(x),) * 3

    monkeypatch.setattr(architecture, "SAM2UNet", ProbeModel)
    csv_path = make_manifest(tmp_path, "probe", 10)
    # Inference should work after removing GT entirely.
    (tmp_path / "probe_mask.png").unlink()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("architecture: sam2_unet_hiera_l\ntrain_manifest: unused.csv\n"
                   "encoder_checkpoint: missing.pt\noutput: unused\ndevice: cpu\n")
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(architecture="sam2_unet_hiera_l", model=ProbeModel().state_dict(),
                    epoch=20), checkpoint)
    out = tmp_path / "prediction"
    result = predict_unet(cfg, checkpoint, csv_path, out, "per-image")
    assert result["count"] == 1
    with (out / "predictions.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["id"] == "probe"
    with Image.open(out / row["prediction"]) as im:
        assert im.size == (17, 15)
    with pytest.raises(FileExistsError):
        predict_unet(cfg, checkpoint, csv_path, out)


def test_rejects_checkpoint_without_official_trunk(tmp_path):
    checkpoint = tmp_path / "wrong.pt"
    torch.save({"model": {"task_tokens": torch.ones(1)}}, checkpoint)
    with pytest.raises(ValueError, match="official SAM2"):
        SAM2UNet(checkpoint, encoder=TinyTrunk())
