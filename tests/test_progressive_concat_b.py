from pathlib import Path

import pytest
import torch

from medsam_research.progressive_concat_b import ProgressiveConcatMambaB
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('size', [11, 22])
def test_native_grids_progressive_inputs_and_gradients(size):
    block = ProgressiveConcatMambaB(d_state=2, expand=1, backend='torch')
    assert [m.kernel_size for m in block.local] == [(1,1),(3,3),(5,5),(7,7)]
    assert len({id(m) for m in block.mamba}) == 4
    seen, previous = [], []
    for m in block.transition:
        m.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    for m in block.mamba:
        m.register_forward_hook(lambda _, args, output: previous.append(output))
    x = torch.randn(1,32,size,size,requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    for i, feature in enumerate(seen):
        torch.testing.assert_close(feature[:,8:], previous[i].transpose(1,2).reshape(1,8,size,size))
    y.square().mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0, name
    with torch.no_grad():
        block.fuse.weight.zero_()
        block.fuse.bias.zero_()
        assert torch.count_nonzero(block(x)) == 0  # no outer residual


def test_model_config_and_training_heads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_pvt_config(root/'configs/polyp_pvt_slots_three_head_concat_b.yaml')
    base = load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    assert {k for k in cfg if cfg[k] != base[k]} == {'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k] != base['slots'].get(k)} == {'b_variant'}
    options = dict(cfg['slots'], d_state=2, expand=1, backend='torch')
    model = PolypPVTSlots(FakeBaseline(), **options)
    assert isinstance(model.b3, ProgressiveConcatMambaB) and isinstance(model.b4, ProgressiveConcatMambaB)
    assert model.b3.mamba[0] is not model.b4.mamba[0]
    x = torch.randn(2,3,64,64)
    out = model(x)
    assert set(out) == {'final','aux3','aux4'}
    slots_loss(out, torch.rand(2,1,64,64), cfg['aux_weights']).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    model.eval()
    with torch.no_grad():
        assert set(model(x)) == {'final'}


@pytest.mark.parametrize('extra', [dict(b=False),dict(b_residual=True),dict(b_scan='axial'),dict(ablation='wo_b')])
def test_reject_incompatible_options(extra):
    with pytest.raises(ValueError):
        validate_slot_options(dict(dict(b=True,b_variant='progressive_concat'),**extra))
