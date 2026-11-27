import copy
from pathlib import Path

import pytest
import torch

from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


def config():
    root=Path(__file__).resolve().parents[1]
    return load_pvt_config(root/'configs/polyp_pvt_slots_three_head_ms_a_no_scale_dual_path.yaml')


def test_topology_only_config_changes():
    root=Path(__file__).resolve().parents[1]
    old=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_ms_a_no_scale.yaml')
    new=config()
    assert {k for k in new if new[k]!=old[k]}=={'slots','output'}
    assert {k for k in new['slots'] if new['slots'][k]!=old['slots'].get(k)}=={'fusion'}
    opts=dict(new['slots'],d_state=2,expand=1,backend='torch')
    current=PolypPVTSlots(FakeBaseline(),**opts)
    prior=PolypPVTSlots(FakeBaseline(),**dict(opts,fusion='aligned'))
    for name in ('cim','a2','b3','b4','SAM'):
        a,b=getattr(current,name),getattr(prior,name)
        assert type(a) is type(b)
        assert {k:v.shape for k,v in a.state_dict().items()}=={k:v.shape for k,v in b.state_dict().items()}
        a.load_state_dict(b.state_dict(),strict=True)
    assert not hasattr(current,'aligned_fusion')


@pytest.mark.parametrize('size',[64,352])
def test_real_grids_heads_and_backprop(size):
    model=PolypPVTSlots(FakeBaseline(),**dict(config()['slots'],d_state=2,expand=1,backend='torch'))
    seen={}
    def watch(name):
        def hook(_,args):seen[name]=[a.detach() for a in args]
        return hook
    for name in ('detail_fusion','context_fusion','SAM','aux3','aux4'):
        getattr(model,name).register_forward_pre_hook(watch(name))
    native={}
    def save(name):
        def hook(_,args,out):native[name]=out.detach()
        return hook
    for name in ('b3','b4'):
        getattr(model,name).register_forward_hook(save(name))
    x=torch.randn(2,3,size,size)
    out=model(x)
    assert set(out)=={'final','aux3','aux4'}
    assert seen['detail_fusion'][0].shape==(2,64,size//4,size//4)
    assert seen['context_fusion'][0].shape==(2,64,size//16,size//16)
    assert all(a.shape==(2,32,size//4,size//4) for a in seen['SAM'])
    torch.testing.assert_close(seen['aux3'][0],native['b3'])
    torch.testing.assert_close(seen['aux4'][0],native['b4'])
    assert all(v.shape==(2,1,size,size) for v in out.values())
    if size==64:
        slots_loss(out,torch.rand(2,1,size,size)).backward()
        for name,p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
        model.eval()
        with torch.no_grad():assert set(model(x))=={'final'}


@pytest.mark.parametrize('extra',[dict(cfm=True),dict(a_f2=False),dict(deep_supervision=False),dict(ablation='wo_a')])
def test_invalid_combination(extra):
    opts=dict(config()['slots']);opts.update(extra)
    with pytest.raises(ValueError):validate_slot_options(opts)


def test_original_abc_on_current_dual_path():
    from medsam_research.abc_decoder import SpatialContextA, LocalMambaB, ChannelSpatialC
    root=Path(__file__).resolve().parents[1]
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_original_abc_dual_path.yaml')
    baseline=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    assert {k for k in cfg if cfg[k]!=baseline[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=baseline['slots'].get(k)}=={'fusion'}
    model=PolypPVTSlots(FakeBaseline(),**dict(cfg['slots'],d_state=2,expand=1,backend='torch'))
    assert type(model.cim) is SpatialContextA and type(model.a2) is SpatialContextA
    assert type(model.b3) is LocalMambaB and type(model.b4) is LocalMambaB
    assert type(model.SAM) is ChannelSpatialC
    assert not model.cim.residual and not model.a2.residual
    assert not model.b3.residual and not model.b4.residual
    seen=[]
    model.SAM.register_forward_pre_hook(lambda _,args:seen.extend(a.shape for a in args))
    x=torch.randn(2,3,64,64)
    out=model(x)
    assert seen==[torch.Size([2,32,16,16])]*2
    assert set(out)=={'final','aux3','aux4'}
    slots_loss(out,torch.rand(2,1,64,64),cfg['aux_weights']).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad():assert set(model(x))=={'final'}
