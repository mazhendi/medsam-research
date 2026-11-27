from pathlib import Path

import pytest
import torch
from torch import nn

from medsam_research.spatial_context_mlp_a import SpatialContextMLPA
from medsam_research.abc_decoder import SpatialContextA, LocalMambaB, ChannelSpatialC
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('channels,size',[(64,88),(128,44)])
def test_original_operations_preserved_and_mlp_residual(channels,size):
    original=SpatialContextA(channels,residual=False)
    block=SpatialContextMLPA(channels)
    missing,unexpected=block.load_state_dict(original.state_dict(),strict=False)
    assert not unexpected and all(k.startswith('mlp.') for k in missing)
    assert len(missing)==4
    assert not any(isinstance(m,nn.LayerNorm) for m in block.modules())
    assert not any('gamma' in k for k in block.state_dict())
    x=torch.randn(1,channels,size,size,requires_grad=True)
    feature=original(x)
    actual=block(x)
    torch.testing.assert_close(actual,feature+block.mlp(feature))
    actual.square().mean().backward()
    for name,p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    with torch.no_grad():
        block.mlp[-1].weight.zero_();block.mlp[-1].bias.zero_()
        torch.testing.assert_close(block(x),original(x),rtol=0,atol=0)
        block.fuse.weight.zero_();block.fuse.bias.zero_()
        assert torch.count_nonzero(block(x))==0 # no input X skip


def test_original_three_head_topology_and_config():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_original_a_mlp.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'a1_variant','a2_variant'}
    model=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],d_state=2,expand=1,backend='torch'))
    assert type(model.cim) is SpatialContextMLPA and type(model.a2) is SpatialContextMLPA
    assert model.cim.mlp[0].weight is not model.a2.mlp[0].weight
    assert type(model.b3) is LocalMambaB and type(model.b4) is LocalMambaB
    assert type(model.SAM) is ChannelSpatialC and model.fusion=='aligned'
    seen=[]
    model.SAM.register_forward_pre_hook(lambda _,args:seen.extend(a.shape for a in args))
    x=torch.randn(2,3,64,64)
    out=model(x)
    assert set(out)=={'final','aux3','aux4'}
    assert seen==[torch.Size([2,32,8,8])]*2
    slots_loss(out,torch.rand(2,1,64,64),cfg['aux_weights']).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad():assert set(model(x))=={'final'}


@pytest.mark.parametrize('options',[
 dict(a1_variant='original_mlp',a=False),
 dict(a1_variant='original_mlp',a=True,a_residual=True),
 dict(a2_variant='original_mlp',a_f2=False)])
def test_invalid_options(options):
    with pytest.raises(ValueError):validate_slot_options(options)
