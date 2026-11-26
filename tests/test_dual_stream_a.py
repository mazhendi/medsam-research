from pathlib import Path

import pytest
import torch

from medsam_research.dual_stream_a import DualStreamGatedA
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.abc_decoder import LocalMambaB, ChannelSpatialC
from medsam_research.multiscale_readwrite_b import MultiscaleReadWriteB
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('channels,size', [(64,88),(128,44)])
def test_native_shapes_formula_backward_and_no_outer_skip(channels,size):
    block=DualStreamGatedA(channels)
    x=torch.randn(1,channels,size,size,requires_grad=True)
    v1,v2=block.project(x).chunk(2,1)
    w1,w2=block.gates(torch.cat((v1,v2),1)).sigmoid().chunk(2,1)
    expected=block.fuse(torch.cat((block.mlp1(v1+w1*v2),block.mlp2(v2+w2*v1)),1))
    actual=block(x)
    torch.testing.assert_close(actual,expected)
    assert actual.shape==x.shape
    actual.square().mean().backward()
    for name,p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    assert block.mlp1[0].weight.data_ptr()!=block.mlp2[0].weight.data_ptr()
    with torch.no_grad():
        block.fuse.weight.zero_();block.fuse.bias.zero_()
        assert torch.count_nonzero(block(x))==0


def test_cross_stream_direction_with_controlled_gates():
    block=DualStreamGatedA(4)
    with torch.no_grad():
        block.project.weight.copy_(torch.eye(4).reshape(4,4,1,1));block.project.bias.zero_()
        block.gates[-1].weight.zero_()
        block.gates[-1].bias.copy_(torch.tensor([80.,80.,-80.,-80.]))
    seen=[]
    block.mlp1.register_forward_pre_hook(lambda _,args:seen.append(args[0]))
    block.mlp2.register_forward_pre_hook(lambda _,args:seen.append(args[0]))
    x=torch.arange(1.,5.).reshape(1,4,1,1)
    block(x)
    a,b=x.chunk(2,1)
    torch.testing.assert_close(seen[0],a+b)
    torch.testing.assert_close(seen[1],b)


@pytest.mark.parametrize('base,new,btype',[
 ('polyp_pvt_slots_three_head','polyp_pvt_slots_three_head_dual_a',LocalMambaB),
 ('polyp_pvt_slots_three_head_readwrite_b','polyp_pvt_slots_three_head_readwrite_b_dual_a',MultiscaleReadWriteB)])
def test_config_and_full_three_head_integration(base,new,btype):
    root=Path(__file__).resolve().parents[1]
    old=load_pvt_config(root/f'configs/{base}.yaml')
    cfg=load_pvt_config(root/f'configs/{new}.yaml')
    assert {k for k in cfg if cfg[k]!=old[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=old['slots'].get(k)}=={'a1_variant','a2_variant'}
    assert cfg['augmentation']=='none' and cfg['aux_weights']==[.5,.5]
    model=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],d_state=2,expand=1,backend='torch'))
    assert isinstance(model.cim,DualStreamGatedA) and isinstance(model.a2,DualStreamGatedA)
    assert isinstance(model.b3,btype) and isinstance(model.b4,btype)
    assert isinstance(model.SAM,ChannelSpatialC)
    x=torch.randn(2,3,64,64)
    out=model(x)
    assert set(out)=={'final','aux3','aux4'}
    slots_loss(out,torch.rand(2,1,64,64),cfg['aux_weights']).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad(): assert set(model(x))=={'final'}


@pytest.mark.parametrize('options',[
 dict(a=False,a1_variant='dual_stream_gated'),
 dict(a=True,a_residual=True,a1_variant='dual_stream_gated'),
 dict(a2_variant='dual_stream_gated'),
])
def test_invalid_options(options):
    with pytest.raises(ValueError):validate_slot_options(options)
