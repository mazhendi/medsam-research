from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.subband_decoder import (
    BCSC,
    SCSI,
    LocalCrossRead,
    SubbandPVT,
    boundary_target,
    decoder_loss,
    haar_merge,
    haar_split,
)


class FakePVT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(3, c, 1) for c in (64, 128, 320, 512))

    def forward(self, x):
        return [layer(F.avg_pool2d(x, stride)) for layer, stride in
                zip(self.layers, (4, 8, 16, 32))]


@pytest.mark.parametrize("shape", [(2, 3, 12, 16), (1, 8, 22, 22)])
def test_haar_roundtrip_energy_and_gradient(shape):
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    bands = haar_split(x)
    reconstructed = haar_merge(bands)
    torch.testing.assert_close(reconstructed, x)
    torch.testing.assert_close(bands.square().sum(), x.square().sum())
    reconstructed.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_haar_constant_has_no_directional_bands():
    bands = haar_split(torch.ones(1, 2, 8, 8))
    assert torch.equal(bands[:, 0], torch.full_like(bands[:, 0], 2))
    assert torch.count_nonzero(bands[:, 1:]) == 0
    with pytest.raises(ValueError, match="even"):
        haar_split(torch.ones(1, 2, 7, 8))


def test_scsi_semantic_conditioning_and_both_projection_gradients():
    torch.manual_seed(4)
    module = SCSI(16)
    x, semantic = torch.randn(2, 16, 12, 16), torch.randn(2, 16, 3, 4)
    output = module(x, semantic)
    assert output.shape == x.shape
    assert not torch.allclose(output, module(x, torch.zeros_like(semantic)))
    output.square().mean().backward()
    for layer in (module.semantic_q, module.semantic_k, module.q, module.k, module.v):
        assert layer.weight.grad is not None and layer.weight.grad.norm() > 0


def test_cross_sampler_offsets_use_source_pixel_units(monkeypatch):
    module = LocalCrossRead(16, 4)
    target, source = torch.randn(1, 16, 4, 6), torch.randn(1, 16, 2, 3)
    grids = []
    original = F.grid_sample

    def capture(x, grid, **kwargs):
        grids.append(grid.detach().clone())
        return original(x, grid, **kwargs)

    monkeypatch.setattr(F, "grid_sample", capture)
    output = module(target, source)
    assert output.shape == target.shape and len(grids) == 9
    yy, xx = torch.meshgrid((torch.arange(4)+.5)*2/4-1,
                            (torch.arange(6)+.5)*2/6-1, indexing="ij")
    center = torch.stack((xx, yy), -1)[None]
    torch.testing.assert_close(grids[4], center)
    torch.testing.assert_close(grids[0]-grids[4],
                               torch.tensor([-2/3, -1.]).expand_as(center))
    output.square().mean().backward()
    assert module.offset[-1].weight.grad.norm() > 0


def test_bcsc_shapes_and_gradients_in_both_directions():
    module = BCSC(16, 4)
    features = [torch.randn(2, 16, n, n, requires_grad=True) for n in (16, 8, 4, 2)]
    output = module(features)
    assert [v.shape for v in output] == [v.shape for v in features]
    sum(v.square().mean() for v in output).backward()
    for direction in (module.fine_reads, module.coarse_reads):
        assert all(m.q.weight.grad.norm() > 0 for m in direction)
    assert all(feature.grad.norm() > 0 for feature in features)


@pytest.mark.parametrize("scsi,bcsc,boundary", [(False, False, True), (True, False, True),
                                              (False, True, True), (True, True, True),
                                              (True, True, False)])
def test_full_decoder_forward_backward_and_ablation_switches(scsi, bcsc, boundary):
    model = SubbandPVT(FakePVT(), width=16, attention_dim=4,
                       scsi=scsi, bcsc=bcsc, boundary=boundary)
    image = torch.randn(2, 3, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > .7).float()
    output = model(image)
    assert set(output) == ({"coarse", "refined", "boundary"} if boundary else {"coarse", "refined"})
    assert all(logits.shape == target.shape for logits in output.values())
    loss = decoder_loss(output, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.layers[0].weight.grad.norm() > 0
    assert model.decoder.coarse_head.weight.grad.norm() > 0
    assert model.decoder.refined_head.weight.grad.norm() > 0
    assert not any(name in dict(model.named_modules()) for name in ("CIM", "CFM", "SAM"))
    if boundary:
        assert model.decoder.edge_head.weight.grad.norm() > 0


def test_boundary_target_and_auxiliary_loss_switch():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 3, 3] = 1
    expected = torch.zeros_like(target)
    expected[..., 2:5, 2:5] = 1
    torch.testing.assert_close(boundary_target(target), expected)
    assert boundary_target(torch.zeros_like(target)).sum() == 0
    outputs = {key: torch.zeros_like(target) for key in ("coarse", "refined", "boundary")}
    assert decoder_loss(outputs, target, .2) > decoder_loss(outputs, target, 0.)


def test_all_proposed_configs_are_fixed_352_with_explicit_controls():
    root = Path(__file__).resolve().parents[1]
    for name in ("full", "control", "scsi_only", "bcsc_only", "no_edge_aux"):
        cfg = load_pvt_config(root / f"configs/subband_pvt_{name}.yaml")
        assert cfg["architecture"] == "subband_pvt_b2" and cfg["image_size"] == 352
        assert cfg["epochs"] == 99 and cfg["batch_size"] == 16
    cfg = load_pvt_config(root / "configs/subband_pvt_no_edge_aux.yaml")
    assert cfg["edge_weight"] == 0 and cfg["decoder"]["boundary"]
