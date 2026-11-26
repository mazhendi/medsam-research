from pathlib import Path

import pytest
import torch

from medsam_research.dual_statistic_b import DualStatisticMambaB, StatisticMambaPath
from medsam_research.abc_decoder import SpatialContextA, ChannelSpatialC
from medsam_research.selective_mamba import SelectiveMamba1
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


@pytest.mark.parametrize('shape',[(3,5),(1,1)])
def test_pooling_matches_valid_neighborhood_reference(shape):
    h,w=shape
    x=-torch.arange(1,h*w+1,dtype=torch.float32).reshape(1,1,h,w)
    average=StatisticMambaPath(1,'average',backend='torch').pool(x)
    maximum=StatisticMambaPath(1,'maximum',backend='torch').pool(x)
    for y in range(h):
        for col in range(w):
            neighborhood=x[:,:,max(0,y-1):min(h,y+2),max(0,col-1):min(w,col+2)]
            torch.testing.assert_close(average[:,:,y,col],neighborhood.mean().reshape(1,1))
            torch.testing.assert_close(maximum[:,:,y,col],neighborhood.max().reshape(1,1))
    assert average.shape==x.shape and maximum.shape==x.shape


def test_paths_independent_and_match_approved_operations():
    block=DualStatisticMambaB(d_state=2,expand=1,backend='torch')
    assert sum(isinstance(m,SelectiveMamba1) for m in block.modules())==2
    assert block.average.mamba is not block.maximum.mamba
    for path in (block.average,block.maximum):
        assert [m.kernel_size for m in path.local]==[(3,3),(5,5),(7,7)]
        assert all(m.groups==32 and m.in_channels==32 for m in path.local)
        assert path.project.in_channels==96 and path.project.out_channels==32
    a={p.data_ptr() for p in block.average.parameters()}
    assert a.isdisjoint({p.data_ptr() for p in block.maximum.parameters()})
    x=torch.randn(1,32,4,5)
    p=block.project(x)
    expected=x+block.fuse(torch.cat((block.average(p),block.maximum(p)),1))
    torch.testing.assert_close(block(x),expected)
    with pytest.raises(ValueError):
        StatisticMambaPath(4,'invalid')


@pytest.mark.parametrize('size',[11,22])
def test_native_grids_all_gradients_and_residual(size):
    block=DualStatisticMambaB(d_state=2,expand=1,backend='torch')
    x=torch.randn(1,32,size,size)
    y=block(x)
    assert y.shape==x.shape
    y.square().mean().backward()
    for name,p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name
    with torch.no_grad():
        block.fuse.weight.zero_();block.fuse.bias.zero_()
        torch.testing.assert_close(block(x),x,rtol=0,atol=0)
        block.residual=False
        assert torch.count_nonzero(block(x))==0


def test_three_heads_original_a_c_and_both_b_replaced():
    model=PolypPVTSlots(FakeBaseline(),a=True,a_f2=True,b=True,c=True,cfm=False,
                       deep_supervision=True,b_variant='dual_statistic',b_residual=True,
                       d_state=2,expand=1,backend='torch')
    assert type(model.b3) is DualStatisticMambaB and type(model.b4) is DualStatisticMambaB
    assert model.b3.average.mamba is not model.b4.average.mamba
    assert type(model.cim) is SpatialContextA and type(model.a2) is SpatialContextA
    assert type(model.SAM) is ChannelSpatialC
    seen=[]
    for head in (model.aux3,model.aux4):
        head.register_forward_pre_hook(lambda _,args:seen.append(args[0].shape[-2:]))
    x=torch.randn(2,3,64,64)
    out=model(x)
    assert set(out)=={'final','aux3','aux4'} and seen==[(4,4),(2,2)]
    slots_loss(out,torch.rand(2,1,64,64)).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad():
        assert set(model(x))=={'final'}
    assert len(seen)==2


def test_config_only_changes_b_and_output():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_dual_stat_b.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'b_variant','b_residual'}
    assert cfg['augmentation']=='none' and cfg['aux_weights']==[.5,.5]


@pytest.mark.parametrize('options',[
    dict(b=False,b_variant='dual_statistic'),
    dict(b=True,b_variant='dual_statistic',b_scan='axial'),
    dict(a=True,a_f2=True,b=True,c=True,deep_supervision=True,cfm=False,
         ablation='wo_b',b_variant='dual_statistic'),
])
def test_invalid_combinations(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)
