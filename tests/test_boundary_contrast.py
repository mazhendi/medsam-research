"""Geometry, data isolation, gradients and checkpoint roundtrip (not accuracy)."""

import csv
import json
from dataclasses import replace

import pytest
import torch
from test_data_metrics import make_manifest
from test_region_shape import FixtureBaseline, FixturePromptEncoder
from torch.nn import functional as F

from medsam_research import boundary_contrast_training as workflow
from medsam_research import engine
from medsam_research.boundary_contrast import (
    BoundaryContrastConfig,
    BoundaryContrastHead,
    BoundaryContrastSAM2,
    InnerOuterBoundary,
    boundary_contrast_loss,
    constrained_pool,
    directional_pairs,
    load_boundary_config,
)
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig


class Prompt(FixturePromptEncoder):
    def forward(self, points=None, boxes=None, masks=None):
        return None, self.no_mask_embed.weight[0][None, :, None, None].expand(1, -1, 4, 4)


class Baseline(FixtureBaseline):
    def __init__(self):
        super().__init__()
        self.sam.sam_prompt_encoder = Prompt()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.sam.num_feature_levels = 3
        self.sam.directly_add_no_mem_embed = False
        self.sam.forward_image = self.encode

    def encode(self, x):
        image = self.sam.image_encoder(x)
        d = self.sam.sam_mask_decoder
        return {
            "backbone_fpn": [
                d.conv_s0(F.avg_pool2d(image, 2)),
                d.conv_s1(F.avg_pool2d(image, 4)),
                F.avg_pool2d(image, 8),
            ]
        }


def test_normal_sign_pair_distances_and_flat_fallback():
    torch.set_num_threads(2)
    x = torch.arange(13).float()[None, None, None].expand(1, 1, 9, 13)
    p = 0.1 + 0.8 * x / 12
    pairs, reliability, normal = directional_pairs(x, p, [1.0, 2.0], 0.05)
    # p increases to the right: outward points left; inner−outer = 2r.
    torch.testing.assert_close(pairs[0, :, 4, 6], torch.tensor([2.0, 4.0]), atol=1e-4, rtol=1e-4)
    assert normal[0, 0, 4, 6] < -0.99
    assert reliability[0, 0, 4, 6] > 0.5
    assert reliability[0, 0, 4, 0] == 0  # no valid pair at the left boundary
    c = BoundaryContrastConfig(width=8)
    module = InnerOuterBoundary(c).eval()
    f = torch.randn(1, 8, 9, 13)
    actual, r, n, gate = module(f, torch.full((1, 1, 9, 13), 0.5))
    torch.testing.assert_close(actual, module.local(f))
    assert r.count_nonzero() == gate.count_nonzero() == n.count_nonzero() == 0


def test_constrained_pool_reduces_cross_boundary_mixing_and_is_differentiable():
    f = torch.zeros(1, 1, 7, 10)
    f[:, :, :, 5:] = 1
    p = f.clone().requires_grad_()
    unconstrained = constrained_pool(f, p, torch.zeros_like(f), 0.0)
    constrained = constrained_pool(f, p, torch.zeros_like(f), 8.0)
    assert constrained[0, 0, 3, 4] < 0.01 * unconstrained[0, 0, 3, 4]
    constrained[:, :, :, :5].sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_head_shape_gradient_paths_and_no_logit_addition():
    torch.manual_seed(8)
    c = BoundaryContrastConfig(width=8)
    head = BoundaryContrastHead([4, 8, 16], c)
    features = [torch.randn(2, n, h, w) for n, h, w in [(4, 16, 20), (8, 8, 10), (16, 4, 5)]]
    out = head(features, (32, 40))
    assert out["logits"].shape == (2, 1, 32, 40)
    target = torch.zeros_like(out["logits"])
    target[:, :, 8:23, 9:29] = 1
    losses = boundary_contrast_loss(out, target, torch.ones_like(target), c)
    losses["total"].backward()
    for parameter in [
        head.coarse.weight,
        head.boundary.contrast[0].weight,
        head.boundary.gate.weight,
        head.fusions[0].boundary_gate.weight,
        head.fusions[0].competition.weight,
        head.decoder[-1].weight,
    ]:
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    # Changing coarse predictions cannot bypass a forced constant final decoder.
    with torch.no_grad():
        head.decoder[-1].weight.zero_()
        head.decoder[-1].bias.fill_(2.0)
        head.coarse.bias.fill_(30.0)
    torch.testing.assert_close(head(features, (32, 40))["logits"], torch.full((2, 1, 32, 40), 2.0))


def test_ablations_and_label_masking():
    torch.manual_seed(3)
    c = BoundaryContrastConfig(width=8)
    head = BoundaryContrastHead([4, 8, 16], c)
    features = [torch.randn(1, n, s, s) for n, s in [(4, 16), (8, 8), (16, 4)]]
    full = head(features, (32, 32))
    head.config = replace(c, mode="equal_fusion")
    for f in head.fusions:
        f.config = head.config
    equal = head(features, (32, 32))
    assert all(torch.all(g == 0.5) for g in equal["fusion_gates"])
    assert not torch.allclose(full["logits"], equal["logits"])
    head.boundary.config = replace(c, mode="no_boundary")
    assert head(features, (32, 32))["boundary"].count_nonzero() == 0
    y, v = torch.zeros(1, 1, 32, 32), torch.ones(1, 1, 32, 32)
    v[:, :, :, 16:] = 0
    changed = y.clone()
    changed[:, :, :, 16:] = 1
    a, b = boundary_contrast_loss(full, y, v, c), boundary_contrast_loss(full, changed, v, c)
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_config_validation(tmp_path):
    path = tmp_path / "mechanism.yaml"
    path.write_text("not_a_setting: 1\n")
    with pytest.raises(ValueError):
        load_boundary_config(path)
    for c in [
        BoundaryContrastConfig(radii=[]),
        BoundaryContrastConfig(normal_scale=0),
        BoundaryContrastConfig(width=0),
        BoundaryContrastConfig(mode="bad"),
    ]:
        with pytest.raises(ValueError):
            c.validate()


def test_train_evaluate_predict_probe_and_checkpoint_binding(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    train = make_manifest(tmp_path, "train.csv", ["aa", "bb"])
    val = make_manifest(tmp_path, "val.csv", ["cc"])
    test = make_manifest(tmp_path, "test.csv", ["dd"])
    source = tmp_path / "pretrained.pt"
    source.write_bytes(b"fixture only")
    cfg = Config(
        ModelConfig(checkpoint=str(source)),
        DataConfig(str(train), str(val)),
        TrainingConfig(
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            amp=False,
            output_dir=str(tmp_path / "base"),
        ),
    )
    monkeypatch.setattr(engine, "build_model", lambda cfg, device: Baseline().to(device))
    baseline = engine.train(cfg) / "best.pt"
    original = baseline.read_bytes()
    out = workflow.train_boundary_contrast(
        cfg, baseline, BoundaryContrastConfig(width=8), tmp_path / "new", epochs=2
    )
    ckpt = out / "last_boundary_contrast.pt"
    restored, _, _ = workflow.build_system(cfg, baseline, checkpoint=ckpt)
    assert all(not p.requires_grad for p in restored.baseline.parameters())
    with pytest.raises(ValueError, match="Not a baseline"):
        engine.load_adapter(Baseline(), ckpt, cfg)
    scores = workflow.evaluate_boundary_contrast(cfg, baseline, ckpt, test, tmp_path / "eval")
    assert scores["boundary_contrast"]["samples"] == 1
    inference = tmp_path / "unlabeled.csv"
    with inference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "image", "mask", "fov"])
        writer.writerow(["new", "dd.png", "nonexistent.png", "nonexistent_fov.png"])
    workflow.predict_boundary_contrast(cfg, baseline, ckpt, inference, tmp_path / "predict")
    assert (tmp_path / "predict/000000.png").is_file()
    changed = tmp_path / "changed.pt"
    changed.write_bytes(b"wrong baseline")
    with pytest.raises(ValueError, match="Different baseline"):
        workflow.build_system(cfg, changed, checkpoint=ckpt)
    probe = workflow.probe_boundary_contrast(
        cfg, baseline, BoundaryContrastConfig(width=8), tmp_path / "probe", steps=3
    )
    report = json.loads((probe / "probe.json").read_text())
    assert report["history"][0]["gradient_norms"]["head.boundary"] > 0
    assert baseline.read_bytes() == original
    assert not list(probe.glob("*.pt"))


@pytest.mark.integration
def test_real_sam2_frozen_encoding_and_reference_equality():
    pytest.importorskip("sam2")
    from sam2.build_sam import build_sam2

    from medsam_research.model import AutoSAM2

    torch.set_num_threads(2)
    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        ckpt_path=None,
        device="cpu",
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.image_size=64"],
    )
    base = AutoSAM2(sam).eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected = base(image)
    model = BoundaryContrastSAM2(base, BoundaryContrastConfig(width=8))
    out = model.forward_details(image)
    torch.testing.assert_close(out["baseline_logits"], expected)
    boundary_contrast_loss(
        out, torch.zeros_like(expected), torch.ones_like(expected), model.config
    )["total"].backward()
    assert all(p.grad is None for p in base.parameters())
    assert model.head.boundary.contrast[0].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        singles = torch.cat([model(x[None]) for x in image])
        torch.testing.assert_close(model(image), singles, atol=1e-4, rtol=1e-4)


def test_cpu_bf16_autocast_keeps_sampling_and_backward_finite():
    torch.set_num_threads(2)
    c = BoundaryContrastConfig(width=8)
    head = BoundaryContrastHead([4, 8, 16], c)
    pyramid = [torch.randn(1, n, z, z) for n, z in [(4, 16), (8, 8), (16, 4)]]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head(pyramid, (32, 32))
        y = torch.zeros(1, 1, 32, 32)
        loss = boundary_contrast_loss(out, y, torch.ones_like(y), c)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
