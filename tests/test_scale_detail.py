"""Check actual neighborhood geometry, candidate constraints and staged gradients."""
import pytest
import torch
from test_region_shape import FixtureBaseline, settings

from medsam_research.region_shape import RegionShapeConfig, RegionShapeSAM2
from medsam_research.scale_detail import AdaptiveScaleSelection, RegionDetailRetrieval


def test_scale_weights_identity_and_controller_gradient_after_opening():
    module = AdaptiveScaleSelection(16, 8)
    x = torch.randn(2, 16, 7, 9)
    output, weights = module(x)
    torch.testing.assert_close(output, x)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 7, 9))
    with torch.no_grad():
        module.output.weight.normal_(std=.02)
    output, _ = module(x)
    output.square().mean().backward()
    assert module.controller.weight.grad.abs().sum() > 0


def test_local_sampling_constant_values_border_and_candidate_permutation():
    module = RegionDetailRetrieval(4, 4, 4, 5)
    s, d = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 8, 8)
    masks, weights = torch.rand(1, 3, 6, 6), torch.tensor([[.2, .3, .5]])
    output, attention = module(s, d, masks, weights)
    torch.testing.assert_close(output, s)
    torch.testing.assert_close(attention.sum(1), torch.ones(1, 4, 4))
    assert attention[0, 0, 0, 0] == 0  # off-image neighbor
    _, permuted = module(s, d, masks.flip(1), weights.flip(1))
    torch.testing.assert_close(attention, permuted)
    # Uniform QK isolates the prior; spatially structured masks must affect it.
    with torch.no_grad():
        module.query.weight.zero_()
        module.query.bias.zero_()
    structured = torch.zeros(1, 1, 8, 8)
    structured[:, :, :, 4:] = 1
    _, constrained = module(s, d, structured, torch.ones(1, 1))
    module.bias_strength = 0
    _, neutral = module(s, d, structured, torch.ones(1, 1))
    assert (constrained-neutral).abs().max() > .001


def test_enabled_system_identity_and_two_step_gradient_flow():
    torch.manual_seed(17)
    base = FixtureBaseline().eval()
    x = torch.rand(2, 3, 32, 32)
    expected = base(x).detach()
    model = RegionShapeSAM2(base, settings(scale_selection=True, detail_retrieval=True))
    torch.testing.assert_close(model(x), expected)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    for step in range(2):
        opt.zero_grad()
        result = model.forward_details(x)
        result['logits'].square().mean().backward()
        for module in (model.scale_module, model.detail_module):
            assert module.output.weight.grad.abs().sum() > 0
        if step == 1:
            assert model.scale_module.controller.weight.grad.abs().sum() > 0
            assert model.detail_module.query.weight.grad.abs().sum() > 0
            assert model.detail_module.key.weight.grad.abs().sum() > 0
        opt.step()
    assert all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize('kwargs', [{'detail_window': 2}, {'detail_bias': -1},
                                    {'scale_selection': 1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RegionShapeConfig(**kwargs).validate()


def test_warm_start_preserves_learned_mechanism(monkeypatch):
    from medsam_research import region_shape_training as workflow
    old = RegionShapeSAM2(FixtureBaseline(), settings()).eval()
    with torch.no_grad():
        old.dense_prompt[-1].weight.normal_(std=.02)
        old.sparse_prompt.weight.normal_(std=.02)
    monkeypatch.setattr(workflow, 'build_system', lambda *a, **kw: (old, None, None))
    new = RegionShapeSAM2(FixtureBaseline(), settings(
        scale_selection=True, detail_retrieval=True)).eval()
    workflow.initialize_feature_extension(new, None, None, 'fixture')
    x = torch.rand(1, 3, 32, 32)
    torch.testing.assert_close(old(x), new(x))


@pytest.mark.integration
def test_real_sam2_feature_extension():
    pytest.importorskip('sam2')
    from sam2.build_sam import build_sam2

    from medsam_research.model import AutoSAM2
    torch.set_num_threads(2)
    sam = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', ckpt_path=None, device='cpu',
                    apply_postprocessing=False, hydra_overrides_extra=['++model.image_size=64'])
    base = AutoSAM2(sam).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = base(x)
    model = RegionShapeSAM2(base, settings(scale_selection=True, detail_retrieval=True))
    output = model(x)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
    output.square().mean().backward()
    assert model.detail_module.output.weight.grad.abs().sum() > 0
    assert model.scale_module.output.weight.grad.abs().sum() > 0
