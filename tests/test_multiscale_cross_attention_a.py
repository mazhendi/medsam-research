from pathlib import Path
import pytest
import torch
from medsam_research.multiscale_cross_attention_a import MultiscaleCrossAttentionA
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.abc_decoder import LocalMambaB, ChannelSpatialC
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('shape', [(3,4),(1,1)])
def test_local_attention_explicit_reference(shape):
    h,w=shape
    module=MultiscaleCrossAttentionA(4,8,width=4,heads=2).double()
    q,k,v=[torch.randn(1,4,h,w,dtype=torch.double,requires_grad=True) for _ in range(3)]
    actual=module.local_attention(q,k,v)
    expected=torch.zeros_like(actual)
    for y in range(h):
        for x in range(w):
            for head in range(2):
                sl=slice(2*head,2*head+2)
                points=[(yy,xx) for yy in range(max(0,y-2),min(h,y+3)) for xx in range(max(0,x-2),min(w,x+3))]
                scores=torch.stack([(q[0,sl,y,x]*k[0,sl,yy,xx]).sum()/2**.5 for yy,xx in points])
                values=torch.stack([v[0,sl,yy,xx] for yy,xx in points])
                expected[0,sl,y,x]=(scores.softmax(0)[:,None]*values).sum(0)
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,v))


@pytest.mark.parametrize('channels,context,width,size',[(64,128,32,88),(128,320,64,44)])
def test_native_grids_and_residual(channels,context,width,size):
    module=MultiscaleCrossAttentionA(channels,context,width)
    x=torch.randn(1,channels,size,size,requires_grad=True)
    g=torch.randn(1,context,size//2,size//2,requires_grad=True)
    out=module(x,g)
    assert out.shape==x.shape
    out.square().mean().backward()
    for name,p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert g.grad.abs().sum()>0
    module.eval()
    with torch.no_grad():
        torch.testing.assert_close(module(x,g),out)
        module.output.weight.zero_();module.output.bias.zero_()
        torch.testing.assert_close(module(x,g),x,rtol=0,atol=0)


def test_three_head_integration_and_config():
    root=Path(__file__).resolve().parents[1]
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_cross_attention_a.yaml')
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'a1_variant','a2_variant'}
    model=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],d_state=2,expand=1,backend='torch'))
    assert type(model.cim) is MultiscaleCrossAttentionA and type(model.a2) is MultiscaleCrossAttentionA
    assert type(model.b3) is LocalMambaB and type(model.b4) is LocalMambaB
    assert type(model.SAM) is ChannelSpatialC and model.fusion=='aligned'
    assert not set(map(id,model.cim.parameters())) & set(map(id,model.a2.parameters()))
    seen=[]
    model.cim.register_forward_pre_hook(lambda _,args:seen.append(tuple(t.shape[1] for t in args)))
    model.a2.register_forward_pre_hook(lambda _,args:seen.append(tuple(t.shape[1] for t in args)))
    x=torch.randn(2,3,64,64)
    out=model(x)
    assert seen==[(64,128),(128,320)]
    assert set(out)=={'final','aux3','aux4'}
    slots_loss(out,torch.rand(2,1,64,64),cfg['aux_weights']).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad():assert set(model(x))=={'final'}
