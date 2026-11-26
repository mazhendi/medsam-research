from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from medsam_research.multiscale_spatial_a import MultiscaleSpatialA, ChannelLayerNorm
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.abc_decoder import LocalMambaB, ChannelSpatialC
from test_polyp_pvt_slots import FakeBaseline


def test_normalization_is_per_pixel_over_channels():
    x=torch.randn(2,8,3,5)
    layer=ChannelLayerNorm(8)
    expected=F.layer_norm(x.permute(0,2,3,1),(8,),layer.norm.weight,layer.norm.bias,layer.norm.eps)
    torch.testing.assert_close(layer(x),expected.permute(0,3,1,2))


@pytest.mark.parametrize('channels,size',[(64,88),(128,44)])
def test_native_grids_full_formula_and_gradients(channels,size):
    block=MultiscaleSpatialA(channels)
    assert [m.kernel_size for m in block.local]==[(3,3),(5,5),(7,7)]
    assert all(m.groups==channels for m in block.local)
    torch.testing.assert_close(block.gamma1,torch.full_like(block.gamma1,.01))
    torch.testing.assert_close(block.gamma2,torch.full_like(block.gamma2,.01))
    x=torch.randn(1,channels,size,size,requires_grad=True)
    p=block.project(block.norm1(x))
    z=x+block.gamma1*block.fuse(torch.cat([m(p) for m in block.local],1))
    expected=z+block.gamma2*block.mlp(block.norm2(z))
    actual=block(x)
    torch.testing.assert_close(actual,expected)
    assert actual.shape==x.shape
    actual.square().mean().backward()
    for name,param in block.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0,name


def test_two_distinct_residual_sources():
    block=MultiscaleSpatialA(8)
    x=torch.randn(1,8,4,5)
    with torch.no_grad():
        block.gamma2.zero_()
        p=block.project(block.norm1(x))
        z=x+block.gamma1*block.fuse(torch.cat([m(p) for m in block.local],1))
        torch.testing.assert_close(block(x),z)
        block.gamma1.zero_()
        torch.testing.assert_close(block(x),x,rtol=0,atol=0)
        block.gamma2.fill_(.01)
        torch.testing.assert_close(block(x),x+block.gamma2*block.mlp(block.norm2(x)))


def test_original_three_head_config_and_integration():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_ms_a.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'a1_variant','a2_variant'}
    assert cfg['augmentation']=='none' and cfg['aux_weights']==[.5,.5]
    model=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],d_state=2,expand=1,backend='torch'))
    assert isinstance(model.cim,MultiscaleSpatialA) and isinstance(model.a2,MultiscaleSpatialA)
    assert isinstance(model.b3,LocalMambaB) and isinstance(model.b4,LocalMambaB)
    assert isinstance(model.SAM,ChannelSpatialC)
    x=torch.randn(2,3,64,64)
    out=model(x)
    assert set(out)=={'final','aux3','aux4'}
    slots_loss(out,torch.rand(2,1,64,64),cfg['aux_weights']).backward()
    for name,param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
    model.eval()
    with torch.no_grad(): assert set(model(x))=={'final'}


@pytest.mark.parametrize('options',[
 dict(a1_variant='multiscale_spatial',a=False),
 dict(a1_variant='multiscale_spatial',a=True,a_residual=True),
 dict(a2_variant='multiscale_spatial',a_f2=False)])
def test_reject_incompatible_options(options):
    with pytest.raises(ValueError):validate_slot_options(options)
