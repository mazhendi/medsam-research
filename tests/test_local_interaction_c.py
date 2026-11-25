from pathlib import Path

import pytest
import torch

from medsam_research.local_interaction_c import LocalCrossAttention, BidirectionalLocalC
from medsam_research.abc_decoder import SpatialContextA, LocalMambaB
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('shape', [(3,4), (1,1), (1,4)])
def test_attention_matches_explicit_valid_neighborhood_reference(shape):
    torch.manual_seed(11)
    block = LocalCrossAttention(4).double()
    h,w = shape
    source, context = torch.randn(1,4,h,w,dtype=torch.float64), torch.randn(1,4,h,w,dtype=torch.float64)
    q,k,v = block.query(source), block.key(context), block.value(context)
    expected = torch.empty_like(q)
    for y in range(h):
        for x in range(w):
            coords = [(yy,xx) for yy in range(max(0,y-1),min(h,y+2))
                      for xx in range(max(0,x-1),min(w,x+2))]
            keys = torch.stack([k[0,:,yy,xx] for yy,xx in coords])
            values = torch.stack([v[0,:,yy,xx] for yy,xx in coords])
            weights = ((keys @ q[0,:,y,x]) / 2).softmax(0)
            expected[0,:,y,x] = (weights[:,None]*values).sum(0)
    torch.testing.assert_close(block(source,context),expected)


def test_border_padding_does_not_dilute_values():
    block = LocalCrossAttention(1)
    with torch.no_grad():
        for projection in (block.query,block.key,block.value):
            for layer in projection:
                layer.weight.zero_()
                layer.bias.zero_()
        block.value[0].weight.fill_(1)
        block.value[1].weight[0,0,1,1] = 1
    actual = block(torch.zeros(1,1,3,3),torch.ones(1,1,3,3))
    torch.testing.assert_close(actual,torch.ones_like(actual))


def test_local_c_real_grid_and_original_selector_normalization():
    block = BidirectionalLocalC(32).eval()
    e,u = torch.randn(1,32,44,44), torch.randn(1,32,22,22)
    with torch.no_grad():
        e_star,u_star,weights = block.prepare(e,u)
        actual = block(e,u)
    assert actual.shape == (1,32,44,44)
    torch.testing.assert_close(weights.sum(1),torch.ones_like(e_star))
    torch.testing.assert_close(actual,block.output(weights[:,0]*e_star+weights[:,1]*u_star))
    assert block.u_to_e.query[0].weight is not block.e_to_u.query[0].weight


def test_candidate_integration_three_heads_gradients_and_eval():
    model = PolypPVTSlots(FakeBaseline(),a=True,a_f2=True,b=True,c=True,
                         cfm=False,deep_supervision=True,c_variant='bidirectional_local',
                         backend='torch',d_state=2,expand=1)
    assert isinstance(model.SAM,BidirectionalLocalC)
    assert type(model.cim) is SpatialContextA and type(model.a2) is SpatialContextA
    assert isinstance(model.b3,LocalMambaB)
    image = torch.randn(2,3,64,64)
    outputs = model(image)
    assert set(outputs) == {'final','aux3','aux4'}
    slots_loss(outputs,torch.rand(2,1,64,64)).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    for direction in (model.SAM.u_to_e,model.SAM.e_to_u):
        for p in direction.parameters():
            assert p.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert set(model(image)) == {'final'}


def test_config_changes_only_c_variant_and_output():
    root = Path(__file__).resolve().parents[1]
    base = load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    new = load_pvt_config(root/'configs/polyp_pvt_slots_three_head_local_c.yaml')
    assert {k for k in new if new[k]!=base[k]} == {'slots','output'}
    assert {k for k in new['slots'] if new['slots'][k]!=base['slots'].get(k)} == {'c_variant'}
    assert new['augmentation'] == 'none' and new['aux_weights'] == [.5,.5]


@pytest.mark.parametrize('options', [
    dict(c=True,c_variant='invalid'), dict(c=False,c_variant='bidirectional_local'),
    dict(a=True,a_f2=True,b=True,c=True,cfm=False,deep_supervision=True,
         ablation='wo_c',c_variant='bidirectional_local'),
])
def test_invalid_c_options(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)
