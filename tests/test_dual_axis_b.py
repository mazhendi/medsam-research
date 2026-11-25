from pathlib import Path

import pytest
import torch
from torch import nn

from medsam_research.dual_axis_b import DualAxisMambaB, TiedBidirectionalMamba
from medsam_research.selective_mamba import SelectiveMamba1
from medsam_research.abc_decoder import SpatialContextA, ChannelSpatialC
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


class CumulativeScan(nn.Module):
    def forward(self,tokens):
        return tokens.cumsum(1)


def test_backward_output_is_aligned_before_fusion():
    block=TiedBidirectionalMamba(2,backend='torch')
    block.mamba=CumulativeScan()
    with torch.no_grad():
        block.projection.weight.zero_();block.projection.bias.zero_()
        # Select forward channel0 and reverse-aligned channel1.
        block.projection.weight[0,0]=1
        block.projection.weight[1,3]=1
    x=torch.arange(1,13,dtype=torch.float32).reshape(2,3,2)
    expected=torch.stack((x[:,:,0].cumsum(1),x[:,:,1].flip(1).cumsum(1).flip(1)),dim=-1)
    torch.testing.assert_close(block(x),expected)


@pytest.mark.parametrize('shape',[(3,5),(1,4),(4,1)])
def test_pack_restore_matches_explicit_independent_row_column_loops(shape):
    block=DualAxisMambaB(4,d_state=2,expand=1,backend='torch').eval()
    block.horizontal.mamba=CumulativeScan();block.vertical.mamba=CumulativeScan()
    h,w=shape
    x=torch.randn(2,4,h,w)
    with torch.no_grad():
        p=block.project(block.norm(x.permute(0,2,3,1)).permute(0,3,1,2))
        horizontal=torch.empty_like(p);vertical=torch.empty_like(p)
        for b in range(2):
            for y in range(h):
                seq=p[b,:,y,:].T.unsqueeze(0)
                horizontal[b,:,y,:]=block.horizontal(seq)[0].T
            for col in range(w):
                seq=p[b,:,:,col].T.unsqueeze(0)
                vertical[b,:,:,col]=block.vertical(seq)[0].T
        expected=x+block.fuse(torch.cat((horizontal,vertical),1))
        torch.testing.assert_close(block(x),expected)


@pytest.mark.parametrize('size',[11,22])
def test_native_grids_gradients_and_outer_identity(size):
    block=DualAxisMambaB(d_state=2,expand=1,backend='torch')
    assert sum(isinstance(m,SelectiveMamba1) for m in block.modules())==2
    assert block.horizontal.mamba is not block.vertical.mamba
    assert not any(isinstance(m,nn.Conv2d) and m.kernel_size!=(1,1) for m in block.modules())
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


def test_bidirectional_real_mamba_stack_matches_independent_calls():
    block=TiedBidirectionalMamba(4,d_state=2,expand=1,backend='torch').eval()
    t=torch.randn(3,7,4)
    with torch.no_grad():
        expected=block.projection(torch.cat((block.mamba(t),block.mamba(t.flip(1)).flip(1)),-1))
        torch.testing.assert_close(block(t),expected)


def test_three_heads_with_both_b_replaced_and_original_a_c():
    model=PolypPVTSlots(FakeBaseline(),a=True,a_f2=True,b=True,c=True,
                       cfm=False,deep_supervision=True,b_variant='dual_axis',b_residual=True,
                       d_state=2,expand=1,backend='torch')
    assert type(model.b3) is DualAxisMambaB and type(model.b4) is DualAxisMambaB
    assert model.b3.horizontal.mamba is not model.b4.horizontal.mamba
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


def test_dual_axis_config_changes_only_b_and_output():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_dual_axis_b.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'b_variant','b_residual'}
    assert cfg['augmentation']=='none' and cfg['aux_weights']==[.5,.5]


@pytest.mark.parametrize('options',[
    dict(b=False,b_variant='dual_axis'),
    dict(b=True,b_variant='dual_axis',b_scan='axial'),
    dict(a=True,a_f2=True,b=True,c=True,deep_supervision=True,cfm=False,
         ablation='wo_b',b_variant='dual_axis'),
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)
