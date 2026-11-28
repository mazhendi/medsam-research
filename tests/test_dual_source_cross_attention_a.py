from pathlib import Path
import torch
from torch.nn import functional as F
from medsam_research.dual_source_cross_attention_a import DualSourceCrossAttentionA
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.abc_decoder import LocalMambaB,ChannelSpatialC
from test_polyp_pvt_slots import FakeBaseline


def test_shared_weights_match_independent_reference_and_gradients():
    m=DualSourceCrossAttentionA(4,8,width=4,heads=2).double()
    tensors=[torch.randn(1,4,3,4,dtype=torch.double,requires_grad=True) for _ in range(4)]
    q,k,vx,vg=tensors
    actual=m.dual_attention(q,k,vx,vg)
    # Parent local attention is independently reference-tested, including boundaries.
    expected=torch.cat([m.local_attention(q,k,vx),m.local_attention(q,k,vg)],1)
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in tensors)


def test_native_grids_and_skip_sources():
    for channels,context,width,size in [(64,128,32,88),(128,320,64,44)]:
        m=DualSourceCrossAttentionA(channels,context,width)
        x=torch.randn(1,channels,size,size,requires_grad=True)
        g=torch.randn(1,context,size//2,size//2,requires_grad=True)
        y=m(x,g)
        assert y.shape==x.shape
        y.square().mean().backward()
        for name,p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
        m.eval()
        with torch.no_grad():
            torch.testing.assert_close(m(x,g),y)
            m.output.weight.zero_();m.output.bias.zero_()
            expected=x+F.interpolate(m.context_skip(g),size=x.shape[-2:],mode='bilinear',align_corners=False)
            torch.testing.assert_close(m(x,g),expected)
            m.context_skip.weight.zero_();m.context_skip.bias.zero_()
            torch.testing.assert_close(m(x,g),x,rtol=0,atol=0)


def test_config_and_three_head_integration():
    root=Path(__file__).resolve().parents[1]
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_dual_source_a.yaml')
    old=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_cross_attention_a.yaml')
    assert {k for k in cfg if cfg[k]!=old[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=old['slots'][k]}=={'a1_variant','a2_variant'}
    m=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],backend='torch',d_state=2,expand=1))
    assert type(m.cim) is DualSourceCrossAttentionA and type(m.a2) is DualSourceCrossAttentionA
    assert not set(map(id,m.cim.parameters())) & set(map(id,m.a2.parameters()))
    assert type(m.b3) is LocalMambaB and type(m.b4) is LocalMambaB
    assert type(m.SAM) is ChannelSpatialC and m.fusion=='aligned'
    x=torch.randn(2,3,64,64)
    out=m(x)
    assert set(out)=={'final','aux3','aux4'}
    slots_loss(out,torch.rand(2,1,64,64),cfg['aux_weights']).backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    # Prediction reconstruction from config/state preserves the changed A exactly.
    restored=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],backend='torch',d_state=2,expand=1))
    restored.load_state_dict(m.state_dict(),strict=True)
    m.eval();restored.eval()
    with torch.no_grad():
        assert set(m(x))=={'final'}
        torch.testing.assert_close(restored(x)['final'],m(x)['final'],rtol=0,atol=0)
