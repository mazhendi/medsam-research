from pathlib import Path

import pytest
import torch

from medsam_research.shared_mamba_b import SharedMambaB
from medsam_research.selective_mamba import SelectiveMamba1
from medsam_research.abc_decoder import SpatialContextA, ChannelSpatialC
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


def test_batch_stack_equals_three_independent_calls_to_same_mamba():
    torch.manual_seed(7)
    block=SharedMambaB(32,d_state=2,expand=1,backend='torch').eval()
    x=torch.randn(2,32,4,5)
    p=block.project(x)
    tokens=[norm(local(p).flatten(2).transpose(1,2)) for local,norm in zip(block.local,block.norm)]
    assert sum(isinstance(m,SelectiveMamba1) for m in block.modules()) == 1
    with torch.no_grad():
        independent=[block.mamba(t) for t in tokens]
        stacked=block.mamba(torch.cat(tokens,0)).chunk(3,0)
        for actual,expected in zip(stacked,independent):
            torch.testing.assert_close(actual,expected)
        expected=x+block.fuse(torch.cat([t.transpose(1,2).reshape_as(x) for t in independent],1))
        torch.testing.assert_close(block(x),expected)
        modified=torch.cat((tokens[0]+10,tokens[1],tokens[2]),0)
        changed=block.mamba(modified).chunk(3,0)
        torch.testing.assert_close(changed[1],stacked[1],rtol=0,atol=0)
        torch.testing.assert_close(changed[2],stacked[2],rtol=0,atol=0)


@pytest.mark.parametrize('size',[11,22])
def test_full_width_paths_gradients_and_identity_residual(size):
    block=SharedMambaB(d_state=2,expand=1,backend='torch')
    assert [m.kernel_size for m in block.local] == [(3,3),(5,5),(7,7)]
    assert all(m.in_channels==32 and m.groups==32 for m in block.local)
    assert block.fuse.in_channels==96
    x=torch.randn(1,32,size,size)
    output=block(x)
    assert output.shape==x.shape
    output.square().mean().backward()
    for name,p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name
    with torch.no_grad():
        block.fuse.weight.zero_();block.fuse.bias.zero_()
        torch.testing.assert_close(block(x),x,rtol=0,atol=0)
        block.residual=False
        assert torch.count_nonzero(block(x))==0


def test_two_deep_blocks_and_three_heads_with_original_a_c():
    model=PolypPVTSlots(FakeBaseline(),a=True,a_f2=True,b=True,c=True,
                       cfm=False,deep_supervision=True,b_variant='shared_mamba',
                       b_residual=True,d_state=2,expand=1,backend='torch')
    assert isinstance(model.b3,SharedMambaB) and isinstance(model.b4,SharedMambaB)
    assert model.b3.mamba is not model.b4.mamba
    assert model.b3.residual and model.b4.residual
    assert type(model.cim) is SpatialContextA and type(model.a2) is SpatialContextA
    assert type(model.SAM) is ChannelSpatialC
    seen=[]
    for head in (model.aux3,model.aux4):
        head.register_forward_pre_hook(lambda _,args:seen.append(args[0].shape[-2:]))
    image=torch.randn(2,3,64,64)
    out=model(image)
    assert set(out)=={'final','aux3','aux4'} and seen==[(4,4),(2,2)]
    slots_loss(out,torch.rand(2,1,64,64)).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad():
        assert set(model(image))=={'final'}
    assert len(seen)==2


def test_config_only_changes_b_variant_outer_residual_and_output():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_shared_b.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'b_variant','b_residual'}
    assert cfg['augmentation']=='none' and cfg['aux_weights']==[.5,.5]


@pytest.mark.parametrize('options',[
    dict(b=False,b_variant='shared_mamba'),
    dict(b=True,b_variant='shared_mamba',b_scan='axial'),
    dict(a=True,a_f2=True,b=True,c=True,deep_supervision=True,cfm=False,
         ablation='wo_b',b_variant='shared_mamba'),
])
def test_invalid_shared_variant_combinations(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)
