"""Native grids, frozen-state persistence, and gradients through fixed modules."""
from dataclasses import replace

import pytest
import torch
from test_region_shape import FixtureBaseline, settings

from medsam_research import region_shape_training as workflow
from medsam_research.multiscale_refinement import CoarseGuidedAttention, MultiScaleRefinement
from medsam_research.region_shape import RegionShapeSAM2


def test_three_grids_identity_and_semantic_dependency():
    block = MultiScaleRefinement([4, 8, 16], width=4)
    inputs = [torch.randn(2, c, h, w) for c, h, w in [(4, 17, 19), (8, 9, 10), (16, 5, 5)]]
    outputs = block(inputs)
    for x, y in zip(inputs, outputs):
        torch.testing.assert_close(x, y)
    # Open outputs to test that coarse K/V actually influence fine queries.
    with torch.no_grad():
        block.middle.output.weight.normal_(std=.1)
        block.fine.output.weight.normal_(std=.1)
    outputs = block(inputs)
    changed = block([inputs[0], inputs[1], torch.randn_like(inputs[2])])
    assert not torch.allclose(outputs[1], changed[1])
    assert not torch.allclose(outputs[0], changed[0])
    assert all(torch.isfinite(y).all() for y in changed)


def test_stage_one_freezes_existing_parameters_but_passes_gradients():
    torch.manual_seed(8)
    model = RegionShapeSAM2(FixtureBaseline(), settings(multiscale_refinement=True,
                                                       refinement_only=True, refinement_width=4))
    assert all(p.requires_grad == name.startswith('refinement.')
               for name, p in model.named_parameters())
    fixed = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    x = torch.randn(2, 3, 32, 32)
    expected = model.baseline(x).detach()
    torch.testing.assert_close(model(x), expected)
    opt = torch.optim.AdamW(model.refinement.parameters(), lr=.001)
    for step in range(2):
        opt.zero_grad()
        model(x).square().mean().backward()
        assert model.refinement.fine.output.weight.grad.abs().sum() > 0
        assert model.refinement.coarse.output.weight.grad.abs().sum() > 0
        if step == 1:
            assert model.refinement.middle.output.weight.grad.abs().sum() > 0
            assert model.refinement.fine.query.weight.grad.abs().sum() > 0
        opt.step()
    for n, p in model.named_parameters():
        if n in fixed:
            torch.testing.assert_close(p, fixed[n], atol=0, rtol=0)
            assert p.grad is None
    # Frozen learned C states and even a warmed decoder are included in export.
    state = workflow.mechanism_state(model)
    assert 'head.queries' in state
    assert 'baseline.sam.sam_mask_decoder.output.weight' in state
    restored = RegionShapeSAM2(FixtureBaseline(), model.config).eval()
    restored.baseline.load_state_dict(model.baseline.state_dict())
    restored.load_state_dict(state, strict=False)
    torch.testing.assert_close(restored(x), model(x))


def test_warm_start_from_trained_rs_preserves_outputs(monkeypatch):
    old = RegionShapeSAM2(FixtureBaseline(), settings()).eval()
    with torch.no_grad():
        old.dense_prompt[-1].weight.normal_(std=.01)
        old.sparse_prompt.weight.normal_(std=.01)
    monkeypatch.setattr(workflow, 'build_system', lambda *a, **kw: (old, None, None))
    new = RegionShapeSAM2(FixtureBaseline(), replace(old.config,
        multiscale_refinement=True, refinement_only=True, refinement_width=4)).eval()
    workflow.initialize_feature_extension(new, None, None, 'fixture')
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(new(x), old(x))


@pytest.mark.integration
def test_real_sam2_enhanced_features_and_frozen_decoder():
    pytest.importorskip('sam2')
    from sam2.build_sam import build_sam2

    from medsam_research.model import AutoSAM2
    torch.set_num_threads(2)
    sam = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', ckpt_path=None, device='cpu',
                    apply_postprocessing=False, hydra_overrides_extra=['++model.image_size=64'])
    model = RegionShapeSAM2(AutoSAM2(sam), settings(multiscale_refinement=True,
                            refinement_only=True, refinement_width=4))
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = model.baseline(x)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().mean().backward()
    assert model.refinement.fine.output.weight.grad.abs().sum() > 0
    assert model.refinement.middle.output.weight.grad.abs().sum() > 0
    assert model.refinement.coarse.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.baseline.parameters())


def test_uniform_attention_preserves_constant_values_at_borders():
    module = CoarseGuidedAttention(2, 2, 2, 3)
    with torch.no_grad():
        module.query.weight.zero_()
        module.query.bias.zero_()
        module.value.weight.zero_()
        module.value.bias.fill_(1)
        module.output.weight.fill_(.5)
    fine = torch.zeros(1, 2, 9, 11)
    actual = module(fine, torch.zeros(1, 2, 3, 4))
    torch.testing.assert_close(actual, torch.ones_like(actual))
