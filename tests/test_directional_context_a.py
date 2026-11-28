from pathlib import Path

import pytest
import torch
from torch import nn

from medsam_research.directional_context_a import DirectionalContextA, DirectionalNeighborhood
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.abc_decoder import LocalMambaB, ChannelSpatialC
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('kernel',[(1,7),(7,1),(3,3)])
@pytest.mark.parametrize('shape',[(3,4),(1,1)])
def test_against_explicit_valid_neighbor_reference(kernel,shape):
    h,w=shape
    layer=DirectionalNeighborhood(4,kernel,heads=2).double()
    with torch.no_grad():layer.bias.normal_()
    q,k,v=[torch.randn(1,4,h,w,dtype=torch.double,requires_grad=True) for _ in range(3)]
    actual=layer(q,k,v)
    expected=torch.zeros_like(actual)
    for y in range(h):
        for x in range(w):
            for head in range(2):
                logits,values=[],[]
                for dy in range(kernel[0]):
                    for dx in range(kernel[1]):
                        yy,xx=y+dy-kernel[0]//2,x+dx-kernel[1]//2
                        if 0<=yy<h and 0<=xx<w:
                            sl=slice(head*2,(head+1)*2)
                            logits.append((q[0,sl,y,x]*k[0,sl,yy,xx]).sum()*layer.scale+layer.bias[head,dy*kernel[1]+dx])
                            values.append(v[0,sl,yy,xx])
                expected[0,head*2:(head+1)*2,y,x]=(torch.stack(logits).softmax(0)[:,None]*torch.stack(values)).sum(0)
    torch.testing.assert_close(actual,expected)
    actual.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,v))


@pytest.mark.parametrize('channels,size',[(64,88),(128,44)])
def test_native_grid_backward_and_topology(channels,size):
    layer=DirectionalContextA(channels)
    assert [m.kernel for m in layer.neighborhoods]==[(1,7),(7,1),(3,3)]
    assert not any(isinstance(m,nn.LayerNorm) for m in layer.modules())
    assert not any('gamma' in k or 'mlp' in k for k in layer.state_dict())
    x=torch.randn(1,channels,size,size)
    y=layer(x)
    assert y.shape==x.shape
    y.square().mean().backward()
    for name,p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    with torch.no_grad():
        layer.fuse.weight.zero_();layer.fuse.bias.zero_()
        assert torch.count_nonzero(layer(x))==0


def test_config_and_original_three_head_integration():
    root=Path(__file__).resolve().parents[1]
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_directional_a.yaml')
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'a1_variant','a2_variant'}
    model=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],d_state=2,expand=1,backend='torch'))
    assert type(model.cim) is DirectionalContextA and type(model.a2) is DirectionalContextA
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
