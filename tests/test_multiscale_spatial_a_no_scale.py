from pathlib import Path

import pytest
import torch

from medsam_research.multiscale_spatial_a import MultiscaleSpatialA, MultiscaleSpatialANoScale
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.abc_decoder import LocalMambaB, ChannelSpatialC
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('channels,size',[(64,88),(128,44)])
def test_direct_residuals_native_grid_and_gradients(channels,size):
    block=MultiscaleSpatialANoScale(channels)
    assert not any('gamma' in k for k in block.state_dict())
    x=torch.randn(1,channels,size,size,requires_grad=True)
    p=block.project(block.norm1(x))
    z=x+block.fuse(torch.cat([m(p) for m in block.local],1))
    expected=z+block.mlp(block.norm2(z))
    actual=block(x)
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward()
    for name,param in block.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0,name


def test_matches_scaled_block_with_unit_gammas():
    old=MultiscaleSpatialA(8,layer_scale_init=1.)
    new=MultiscaleSpatialANoScale(8)
    new.load_state_dict({k:v for k,v in old.state_dict().items() if not k.startswith('gamma')},strict=True)
    x=torch.randn(2,8,4,5)
    torch.testing.assert_close(new(x),old(x),rtol=0,atol=0)
    assert sum(p.numel() for p in old.parameters())-sum(p.numel() for p in new.parameters())==16


def test_three_heads_original_b_c_and_config_isolation():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_ms_a.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_ms_a_no_scale.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'a1_variant','a2_variant'}
    model=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],d_state=2,expand=1,backend='torch'))
    assert type(model.cim) is MultiscaleSpatialANoScale and type(model.a2) is MultiscaleSpatialANoScale
    assert isinstance(model.b3,LocalMambaB) and isinstance(model.SAM,ChannelSpatialC)
    x=torch.randn(2,3,64,64)
    out=model(x)
    assert set(out)=={'final','aux3','aux4'}
    slots_loss(out,torch.rand(2,1,64,64),cfg['aux_weights']).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad():assert set(model(x))=={'final'}
