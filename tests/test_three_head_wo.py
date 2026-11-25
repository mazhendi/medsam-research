from pathlib import Path

import pytest
import torch
from torch import nn

from medsam_research.abc_decoder import ChannelSpatialC, LocalMambaB, SpatialContextA
from medsam_research.polyp_pvt_slots import PolypPVTSlots, PlainConcatC, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


def make_model(ablation):
    return PolypPVTSlots(FakeBaseline(), a=True, a_f2=True, b=True, c=True,
                        cfm=False, deep_supervision=True, ablation=ablation,
                        backend='torch', d_state=2, expand=1)


@pytest.mark.parametrize('ablation', ['wo_a', 'wo_b', 'wo_c'])
def test_deletion_graph_heads_and_gradients(ablation):
    model = make_model(ablation)
    if ablation == 'wo_a':
        assert not any(isinstance(m, SpatialContextA) for m in model.modules())
        assert not any(hasattr(model, k) for k in ('cim', 'a2', 'ca', 'sa'))
        assert isinstance(model.SAM, ChannelSpatialC)
    elif ablation == 'wo_b':
        assert isinstance(model.b3, nn.Identity) and isinstance(model.b4, nn.Identity)
        assert not any(isinstance(m, LocalMambaB) for m in model.modules())
        assert isinstance(model.cim, SpatialContextA) and isinstance(model.SAM, ChannelSpatialC)
    else:
        assert isinstance(model.SAM, PlainConcatC)
        assert not any(isinstance(m, ChannelSpatialC) for m in model.modules())
        assert isinstance(model.cim, SpatialContextA) and isinstance(model.b3, LocalMambaB)
    assert not hasattr(model, 'CFM')
    seen = []
    model.aux3.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[-2:]))
    model.aux4.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[-2:]))
    image = torch.randn(2, 3, 64, 64)
    outputs = model(image)
    assert set(outputs) == {'final', 'aux3', 'aux4'}
    assert seen == [(4,4), (2,2)]
    assert all(x.shape == (2,1,64,64) for x in outputs.values())
    slots_loss(outputs, torch.rand(2,1,64,64)).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    model.eval()
    with torch.no_grad():
        assert set(model(image)) == {'final'}
    assert len(seen) == 2


def test_plain_c_preserves_both_streams_and_concat_order():
    module = PlainConcatC(2)
    fine, deep = torch.randn(1,2,8,8), torch.randn(1,2,4,4)
    expected = module.projection(torch.cat((fine, torch.nn.functional.interpolate(
        deep, (8,8), mode='bilinear', align_corners=True)), 1))
    torch.testing.assert_close(module(fine,deep), expected)
    assert list(module.children()) == [module.projection]


def test_wo_a_fine_projection_receives_raw_f1():
    model = make_model('wo_a').eval()
    image = torch.randn(1,3,64,64)
    seen = []
    model.Translayer2_0.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    with torch.no_grad():
        f1 = model.backbone(image)[0]
        model(image)
    torch.testing.assert_close(seen[0], f1, rtol=0, atol=0)


def test_wo_configs_only_change_ablation_and_output():
    root = Path(__file__).resolve().parents[1]
    base = load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    outputs = {base['output']}
    for tag in ('a','b','c'):
        cfg = load_pvt_config(root/f'configs/polyp_pvt_slots_three_head_wo_{tag}.yaml')
        assert {k for k in cfg if cfg[k] != base[k]} == {'slots','output'}
        assert {k for k in cfg['slots'] if cfg['slots'][k] != base['slots'].get(k)} == {'ablation'}
        assert cfg['slots']['ablation'] == f'wo_{tag}'
        assert cfg['augmentation'] == 'none' and cfg['aux_weights'] == [.5,.5]
        outputs.add(cfg['output'])
    assert len(outputs) == 4


@pytest.mark.parametrize('options', [
    {'ablation': 'unknown'}, {'ablation': 'wo_a'},
    dict(a=True,a_f2=True,b=True,c=True,deep_supervision=True,cfm=True,ablation='wo_c'),
])
def test_invalid_deletion_options_rejected(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)


def test_full_default_has_exact_checkpoint_and_output_parity():
    default = make_model('none').eval()
    legacy = PolypPVTSlots(FakeBaseline(), a=True,a_f2=True,b=True,c=True,
                          cfm=False,deep_supervision=True,backend='torch',d_state=2,expand=1).eval()
    legacy.load_state_dict(default.state_dict(), strict=True)
    image = torch.randn(1,3,64,64)
    with torch.no_grad():
        torch.testing.assert_close(default(image)['final'],legacy(image)['final'],rtol=0,atol=0)
