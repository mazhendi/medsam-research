from pathlib import Path

import pytest
import torch

from medsam_research.balanced_star_a import BalancedStarA, StarChannelInteraction
from medsam_research.abc_decoder import SpatialContextA
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


def test_star_operator_matches_independent_transforms():
    block = StarChannelInteraction(8)
    x = torch.randn(2, 8, 7, 9)
    expected = block.output(block.first(x).clamp(0, 6) * block.second(x))
    torch.testing.assert_close(block(x), expected)
    assert block.first.weight is not block.second.weight


def test_balanced_branches_match_diagram_and_backpropagate_at_a1_grid():
    block = BalancedStarA(64)
    x = torch.randn(1, 64, 88, 88, requires_grad=True)
    p = block.project(x)
    parts = []
    for branch, sizes in ((block.spatial, (3, 5)), (block.context, (5, 7))):
        assert tuple(layer.kernel_size[0] for layer in branch.filters) == sizes
        h = sum(layer(p) for layer in branch.filters) / 2
        parts.append(branch.interaction(h).sigmoid() * p)
    assert block.spatial.filters[1].weight is not block.context.filters[0].weight
    expected = block.fuse(torch.cat(parts, 1))
    actual = block(x)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    actual.square().mean().backward()
    for parameter in block.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_only_a1_replaced_and_three_heads_remain_train_only():
    model = PolypPVTSlots(FakeBaseline(), a=True, a_f2=True, b=True, c=True,
                         cfm=False, deep_supervision=True, a1_variant="balanced_star",
                         d_state=2, expand=1, backend="torch")
    assert isinstance(model.cim, BalancedStarA)
    assert type(model.a2) is SpatialContextA
    seen = []
    for head in (model.aux3, model.aux4):
        head.register_forward_hook(lambda *_: seen.append(True))
    image = torch.randn(2, 3, 64, 64)
    outputs = model(image)
    assert set(outputs) == {"final", "aux3", "aux4"}
    assert all(x.shape == (2, 1, 64, 64) for x in outputs.values())
    slots_loss(outputs, torch.rand(2, 1, 64, 64)).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert len(seen) == 2
    model.eval()
    with torch.no_grad():
        assert set(model(image)) == {"final"}
    assert len(seen) == 2


def test_new_config_changes_only_a1_and_output():
    root = Path(__file__).resolve().parents[1]
    original = load_pvt_config(root / "configs/polyp_pvt_slots_three_head.yaml")
    new = load_pvt_config(root / "configs/polyp_pvt_slots_three_head_a1_star.yaml")
    assert {k for k in new if new[k] != original[k]} == {"slots", "output"}
    assert {k for k in new["slots"] if new["slots"][k] != original["slots"].get(k)} == {"a1_variant"}
    assert new["augmentation"] == "none"
    assert new["aux_weights"] == [.5, .5]
    assert new["scales"] == [.75, 1., 1.25]


@pytest.mark.parametrize("options", [
    {"a": True, "a1_variant": "invalid"},
    {"a": False, "a1_variant": "balanced_star"},
    {"a": True, "a_residual": True, "a1_variant": "balanced_star"},
])
def test_invalid_a1_options_rejected(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)


def test_original_default_retains_exact_checkpoint_and_outputs():
    model = PolypPVTSlots(FakeBaseline(), a=True, a_f2=True, cfm=False,
                         deep_supervision=True).eval()
    explicit = PolypPVTSlots(FakeBaseline(), a=True, a_f2=True, cfm=False,
                            deep_supervision=True, a1_variant="original").eval()
    explicit.load_state_dict(model.state_dict(), strict=True)
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(image)["final"], explicit(image)["final"], rtol=0, atol=0)


def test_both_a_blocks_replaced_with_independent_weights_and_gradients():
    model = PolypPVTSlots(FakeBaseline(), a=True, a_f2=True, b=True, c=True,
                         cfm=False, deep_supervision=True, a1_variant="balanced_star",
                         a2_variant="balanced_star", d_state=2, expand=1, backend="torch")
    assert isinstance(model.cim, BalancedStarA) and isinstance(model.a2, BalancedStarA)
    assert model.cim.project.in_channels == 64
    assert model.a2.project.in_channels == 128
    outputs = model(torch.randn(2, 3, 64, 64))
    assert set(outputs) == {"final", "aux3", "aux4"}
    slots_loss(outputs, torch.rand(2, 1, 64, 64)).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    model.eval()
    with torch.no_grad():
        assert set(model(torch.randn(1, 3, 64, 64))) == {"final"}


def test_dual_a_config_only_changes_a2_and_output_from_a1_experiment():
    root = Path(__file__).resolve().parents[1]
    old = load_pvt_config(root / "configs/polyp_pvt_slots_three_head_a1_star.yaml")
    new = load_pvt_config(root / "configs/polyp_pvt_slots_three_head_a12_star.yaml")
    assert {k for k in new if new[k] != old[k]} == {"slots", "output"}
    assert {k for k in new["slots"] if new["slots"][k] != old["slots"].get(k)} == {"a2_variant"}
    assert new["slots"]["a2_variant"] == "balanced_star"
    assert new["augmentation"] == "none" and new["aux_weights"] == [.5, .5]


@pytest.mark.parametrize("options", [
    {"a": True, "a_f2": True, "a2_variant": "invalid"},
    {"a": True, "a_f2": False, "a2_variant": "balanced_star"},
    {"a": True, "a_f2": True, "a_residual": True, "a2_variant": "balanced_star"},
])
def test_invalid_a2_options_rejected(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)
