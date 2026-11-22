from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.scale_direction_decoder import (
    ADF,
    RTR,
    SDS,
    ScaleDirectionPVT,
    probability_geometry,
    sample_along_normal,
    scale_direction_loss,
)


class FakePVT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(3, c, 1) for c in (64, 128, 320, 512))

    def forward(self, x):
        return [layer(F.avg_pool2d(x, stride)) for layer, stride in
                zip(self.layers, (4, 8, 16, 32))]


def test_sds_weights_are_per_position_and_semantic_conditioning_has_gradient():
    torch.manual_seed(3)
    module = SDS(16)
    x, context = torch.randn(2, 16, 12, 16), torch.randn(2, 16, 3, 4, requires_grad=True)
    weights = module.spatial_weights(x, context)
    assert weights.shape == (2, 5, 12, 16)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 12, 16))
    assert not torch.allclose(weights[..., 0, 0], weights[..., 4, 4])
    output = module(x, context)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert context.grad.norm() > 0
    assert all(branch.weight.grad.norm() > 0 for branch in module.branches)


def test_adf_relations_and_both_evidence_paths():
    x = torch.randn(2, 16, 12, 16)
    agree, disagree = ADF.relation_weights(x, x)
    torch.testing.assert_close(agree, torch.ones_like(agree))
    torch.testing.assert_close(disagree, torch.zeros_like(disagree))
    agree, disagree = ADF.relation_weights(x, -x)
    torch.testing.assert_close(agree, torch.zeros_like(agree))
    torch.testing.assert_close(disagree, torch.ones_like(disagree))
    a, d = ADF.relation_weights(torch.zeros_like(x), torch.zeros_like(x))
    assert torch.isfinite(a+d).all()
    module = ADF(16)
    output = module(x, torch.randn(2, 16, 3, 4))
    assert output.shape == x.shape
    output.square().mean().backward()
    assert module.agreement[0].weight.grad.norm() > 0
    assert module.disagreement[0].weight.grad.norm() > 0


def test_flat_probability_has_zero_gate_and_finite_geometry_gradients():
    p = torch.full((2, 1, 8, 10), .5, requires_grad=True)
    normal, gate = probability_geometry(p)
    assert torch.equal(normal, torch.zeros_like(normal))
    assert torch.equal(gate, torch.zeros_like(gate))
    (normal.square().mean()+gate.mean()).backward()
    assert torch.isfinite(p.grad).all()


def test_gradient_direction_and_sampling_are_in_feature_pixels():
    ramp = torch.arange(9, dtype=torch.float32).view(1, 1, 1, 9).expand(1, 1, 7, 9)
    normal, gate = probability_geometry(ramp/9)
    torch.testing.assert_close(normal[:, 0, 2:-2, 2:-2], torch.ones(1, 3, 5))
    assert torch.count_nonzero(normal[:, 1]) == 0
    assert ((gate > 0) & (gate < 1)).all()
    horizontal = torch.cat((torch.ones_like(ramp), torch.zeros_like(ramp)), 1)
    plus, minus = sample_along_normal(ramp, horizontal, 2), sample_along_normal(ramp, horizontal, -2)
    torch.testing.assert_close((plus-minus)[..., 2:-2], torch.full((1, 1, 7, 5), 4.))
    torch.testing.assert_close(sample_along_normal(ramp, horizontal, 0), ramp)


def test_rtr_flat_fallback_is_independent_of_contrast_branch():
    module = RTR(16)
    region, fine = torch.randn(2, 16, 8, 10), torch.randn(2, 16, 8, 10)
    p = torch.full((2, 1, 8, 10), .5)
    before = module(region, fine, p)
    before.square().mean().backward()
    assert module.local.weight.grad.norm() > 0
    with torch.no_grad():
        module.contrast_projection.weight.fill_(10)
        module.contrast_projection.bias.fill_(20)
    torch.testing.assert_close(module(region, fine, p), before)


def test_rtr_nonflat_sampling_path_receives_finite_gradients():
    module = RTR(16)
    region, fine = torch.randn(2, 16, 8, 10), torch.randn(2, 16, 8, 10)
    logits = torch.randn(2, 1, 8, 10, requires_grad=True)
    output = module(region, fine, logits.sigmoid())
    assert output.shape == fine.shape
    output.square().mean().backward()
    assert module.contrast_projection.weight.grad.norm() > 0
    assert logits.grad.norm() > 0 and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("sds,adf,rtr", [(False,False,False),(True,False,False),
    (False,True,False),(False,False,True),(True,True,False),(True,False,True),
    (False,True,True),(True,True,True)])
def test_all_module_combinations_forward_backward(sds, adf, rtr):
    model = ScaleDirectionPVT(FakePVT(), width=16, sds=sds, adf=adf, rtr=rtr)
    outputs = model(torch.randn(2, 3, 64, 64))
    assert set(outputs) == {"final", "coarse", "aux2", "aux3"}
    assert all(value.shape == (2, 1, 64, 64) for value in outputs.values())
    loss = scale_direction_loss(outputs, (torch.rand(2, 1, 64, 64)>.7).float())
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.final_head.weight.grad.norm() > 0
    assert model.backbone.layers[0].weight.grad.norm() > 0


def test_configs_have_same_protocol_and_unique_outputs():
    root = Path(__file__).resolve().parents[1]
    names = ("full", "control", "sds_only", "adf_only", "rtr_only", "sds_adf", "sds_rtr", "adf_rtr")
    configs = [load_pvt_config(root / f"configs/sds_adf_rtr_{name}.yaml") for name in names]
    assert len({cfg["output"] for cfg in configs}) == 8
    for cfg in configs:
        assert cfg["architecture"] == "sds_adf_rtr_pvt_b2"
        assert cfg["image_size"] == 352 and cfg["epochs"] == 99
        assert cfg["batch_size"] == 16 and cfg["decoder"]["tau"] == .02
        assert cfg["scales"] == [.75, 1., 1.25]
