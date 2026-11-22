import csv
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from torch import nn
from torch.utils.data import DataLoader

from medsam_research.data import read_manifest
from medsam_research.local_background import (
    LocalBackgroundDataset,
    background_boxes,
    background_loss,
    local_training_step,
    positive_box,
    select_background,
)
from medsam_research.local_background_workflow import load_local_config
from medsam_research.sam2_unet_workflow import structure_loss


def records(tmp_path, full=False):
    mask = np.zeros((64, 80), np.uint8)
    mask[22:35, 29:43] = 255
    if full:
        mask[:] = 255
    image = np.stack([mask, np.arange(80, dtype=np.uint8)[None].repeat(64, 0), mask], -1)
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    path = tmp_path / "train.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "mask"])
        writer.writeheader()
        writer.writerow(dict(id="case", image="image.png", mask="mask.png"))
    return read_manifest(path)


def test_crops_preserve_foreground_and_exclude_protected_mask():
    mask = np.zeros((96, 120), np.uint8)
    mask[35:60, 45:70] = 1
    for seed in range(20):
        box = positive_box(mask, random.Random(seed))
        x0, y0, x1, y1 = box
        assert mask[y0:y1, x0:x1].sum() == mask.sum()
        boxes = background_boxes(mask, random.Random(seed), margin_fraction=.02)
        protected = maximum_filter(mask, size=5) > 0
        assert len(boxes) == 4
        for x0, y0, x1, y1 in boxes:
            assert not protected[y0:y1, x0:x1].any()
    assert background_boxes(np.ones((64, 64), np.uint8), random.Random(0)) == []


def test_dataset_pairing_determinism_and_safe_fallback(tmp_path):
    dataset = LocalBackgroundDataset(records(tmp_path))
    a, b = dataset[0], dataset[0]
    assert all(torch.equal(a[key], b[key]) for key in a if torch.is_tensor(a[key]))
    assert a["available"] and a["background_gt"].count_nonzero() == 0
    assert a["positive_gt"].max() >= .5
    assert a["candidates"].shape == (4, 3, 352, 352)
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    fallback = LocalBackgroundDataset(records(fallback_dir, full=True))[0]
    assert not fallback["available"]
    assert torch.equal(fallback["background_gt"], fallback["full_gt"])
    assert fallback["background_gt"].min() == 1
    assert torch.equal(fallback["candidates"][0], fallback["full"])


class Spy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        nn.init.constant_(self.conv.weight, .1)
        nn.init.zeros_(self.conv.bias)
        self.calls = []

    def forward(self, x):
        self.calls.append((torch.is_grad_enabled(), self.training))
        return self.conv(x)


def test_hard_selection_uses_current_model_and_preserves_train_mode():
    model = Spy().train()
    candidates = torch.stack([torch.full((2, 3, 8, 8), float(k)) for k in range(4)], 1)
    chosen, _ = select_background(model, candidates, "hard", True)
    assert torch.equal(chosen, candidates[:, 3])
    assert model.training
    assert model.calls == [(False, False)] * 4
    chosen, _ = select_background(model, candidates, "random", True)
    assert torch.equal(chosen, candidates[:, 0])
    chosen, _ = select_background(model, candidates, "hard", False)
    assert torch.equal(chosen, candidates[:, 0])


@pytest.mark.parametrize("mode", ["random", "hard", "matched_full"])
def test_all_arms_match_forward_backward_and_optimizer_budget(mode):
    model = Spy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    batch = dict(full=torch.randn(2, 3, 16, 16), full_gt=torch.ones(2, 1, 16, 16),
                 positive=torch.randn(2, 3, 16, 16), positive_gt=torch.ones(2, 1, 16, 16),
                 candidates=torch.randn(2, 4, 3, 16, 16),
                 available=torch.tensor([True, False]), background_gt=torch.zeros(2, 1, 16, 16))
    batch["background_gt"][1] = 1  # Explicit fallback uses true full-image target.
    before = model.conv.weight.detach().clone()
    report = local_training_step(model, batch, optimizer, mode, True)
    assert report["forward_calls"] == 7 and report["backward_calls"] == 3
    assert report["background_fallbacks"] == 1
    assert model.calls == [(False, False)] * 4 + [(True, True)] * 3
    assert not torch.equal(before, model.conv.weight)
    assert torch.isfinite(model.conv.weight.grad).all()


def test_background_loss_rejects_wrong_negative_labels_and_handles_fallback():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.ones_like(logits)
    with pytest.raises(ValueError, match="zero masks"):
        background_loss(logits, target, torch.tensor([True, False]))
    assert torch.allclose(background_loss(logits, target, torch.tensor([False, False])),
                          structure_loss(logits, target))
    target = torch.zeros_like(logits)
    loss = background_loss(logits, target, torch.tensor([True, True]))
    loss.backward()
    assert (logits.grad > 0).all()  # Gradient descent suppresses predicted foreground.


def test_local_configs_only_change_selection_and_output():
    root = Path(__file__).resolve().parents[1] / "configs"
    configs = [load_local_config(root / f"native_local_{mode}.yaml")
               for mode in ("random", "hard", "matched_full")]
    for cfg in configs:
        assert not any(cfg["modules"].values())
        assert cfg["batch_size"] == 16 and cfg["epochs"] == 20
        cfg.pop("output")
        cfg["local_background"].pop("mode")
    assert configs[0] == configs[1] == configs[2]


def test_loader_preserves_available_bool_dtype(tmp_path):
    batch = next(iter(DataLoader(LocalBackgroundDataset(records(tmp_path)), batch_size=1)))
    assert batch["available"].dtype == torch.bool


@pytest.mark.integration
def test_real_native_decoder_accepts_three_view_update(tmp_path):
    pytest.importorskip("sam2")
    from medsam_research.native_sam2 import build_native_model

    torch.set_num_threads(2)
    model = build_native_model(dict(sam_config="configs/sam2/sam2_hiera_t.yaml",
                                    modules=dict(scds=False, sbca=False, rcsr=False)),
                               allow_random=True)
    batch = next(iter(DataLoader(LocalBackgroundDataset(records(tmp_path)), batch_size=1)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    result = local_training_step(model, batch, optimizer, "hard", True)
    assert result["backward_calls"] == 3 and result["background_count"] == 1
    assert model.encoder.blocks[0].prompt_learn[0].weight.grad.abs().sum() > 0
    assert model.decoder.output_hypernetworks_mlps[0].layers[0].weight.grad.abs().sum() > 0
