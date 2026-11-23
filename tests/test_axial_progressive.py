import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from medsam_research.axial_progressive import axial_bidirectional_mix, ProgressiveFusion
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


def test_axial_sequences_reset_per_row_column_and_restore_reverse_order():
    class Prefix(nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []
        def forward(self, x):
            self.shapes.append(tuple(x.shape))
            return x.cumsum(1)
    x = torch.arange(1., 13.).reshape(1, 2, 2, 3)
    mixer = Prefix()
    actual = axial_bidirectional_mix(x, nn.Identity(), mixer)
    expected = (x.cumsum(3) + x.flip(3).cumsum(3).flip(3)
                + x.cumsum(2) + x.flip(2).cumsum(2).flip(2)) / 4
    torch.testing.assert_close(actual, expected)
    assert mixer.shapes == [(2, 3, 2)] * 2 + [(3, 2, 2)] * 2


@pytest.mark.parametrize('scan,fusion', [('axial','aligned'), ('raster','progressive'), ('axial','progressive')])
def test_variants_keep_native_auxiliary_taps_and_all_parameter_gradients(scan, fusion):
    model = PolypPVTSlots(FakeBaseline(), a=True, a_f2=True, b=True, c=True,
                         cfm=False, deep_supervision=True, b_scan=scan, fusion=fusion,
                         d_state=2, expand=1, backend='torch')
    seen = []
    for head in (model.aux3, model.aux4):
        head.register_forward_pre_hook(lambda _, args: seen.append(tuple(args[0].shape)))
    x = torch.randn(2,3,64,96)
    out = model(x)
    assert seen == [(2,32,4,6), (2,32,2,3)]
    assert all(y.shape == (2,1,64,96) for y in out.values())
    slots_loss(out, (torch.rand(2,1,64,96)>.7).float()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    seen.clear()
    with torch.inference_mode():
        assert set(model.eval()(x)) == {'final'}
    assert seen == []


def test_progressive_fusion_routes_middle_output_into_fine_stage():
    module = ProgressiveFusion()
    observed = []
    middle_output = []
    module.j3.register_forward_hook(lambda _, args, out: middle_output.append(out))
    module.j2.register_forward_pre_hook(lambda _, args: observed.append(args))
    fine = torch.randn(2,32,8,12)
    out = module(torch.randn(2,32,2,3), torch.randn(2,32,4,6), fine)
    assert observed[0][0] is middle_output[0]
    assert observed[0][1] is fine
    assert out.shape == fine.shape
    assert module.j3 is not module.j2


def test_explicit_legacy_switches_preserve_old_checkpoint_and_output():
    options = dict(a=True,a_f2=True,b=True,c=True,cfm=False,deep_supervision=True,
                   d_state=2,expand=1,backend='torch')
    baseline = FakeBaseline()
    original = PolypPVTSlots(copy.deepcopy(baseline), **options).eval()
    explicit = PolypPVTSlots(copy.deepcopy(baseline), **options,b_scan='raster',fusion='aligned').eval()
    explicit.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2,3,64,64)
    with torch.inference_mode():
        torch.testing.assert_close(original(x)['final'], explicit(x)['final'],rtol=0,atol=0)


@pytest.mark.parametrize('options', [dict(b_scan='four'),dict(fusion='sum'),
                                    dict(b_scan='axial'),dict(fusion='progressive')])
def test_invalid_variant_switches(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)


def test_new_configs_change_only_named_model_switches_and_output():
    root=Path(__file__).resolve().parents[1]
    old=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    outputs={old['output']}
    for name,scan,fusion in [('axial','axial','aligned'),('progressive','raster','progressive'),
                             ('axial_progressive','axial','progressive')]:
        new=load_pvt_config(root/f'configs/polyp_pvt_slots_three_head_{name}.yaml')
        assert {k for k in old if old[k]!=new[k]} == {'slots','output'}
        assert new['slots'] == dict(old['slots'],b_scan=scan,fusion=fusion)
        outputs.add(new['output'])
    assert len(outputs)==4
