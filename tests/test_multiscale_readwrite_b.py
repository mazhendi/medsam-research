from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from medsam_research.multiscale_readwrite_b import ReadWriteConditionedMamba, MultiscaleReadWriteB
from medsam_research.abc_decoder import SpatialContextA, ChannelSpatialC
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


def test_scan_matches_explicit_recurrence_and_input_gradients():
    torch.manual_seed(17)
    block=ReadWriteConditionedMamba(4,d_state=2,expand=1,backend='torch').double()
    tokens=torch.randn(2,7,4,dtype=torch.float64,requires_grad=True)
    condition=torch.randn(2,7,4,dtype=torch.float64,requires_grad=True)
    u,z,delta,b,c=block.scan_inputs(tokens,condition)
    a=-block.A_log.exp()
    step=F.softplus(delta+block.dt_proj.bias[None,:,None])
    h=torch.zeros(2,block.inner,block.state_size,dtype=torch.float64)
    outputs=[]
    for t in range(7):
        h=torch.exp(step[:,:,t,None]*a[None])*h+step[:,:,t,None]*b[:,None,:,t]*u[:,:,t,None]
        outputs.append((h*c[:,None,:,t]).sum(-1)+block.D*u[:,:,t])
    expected=block.out_proj(torch.stack(outputs,1)*F.silu(z))
    actual=block(tokens,condition)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-11)
    grads=torch.autograd.grad(actual.square().sum(),(tokens,condition),retain_graph=True)
    reference=torch.autograd.grad(expected.square().sum(),(tokens,condition))
    for result,target in zip(grads,reference):
        torch.testing.assert_close(result,target,rtol=1e-8,atol=1e-10)


def test_condition_changes_only_read_write_not_content_step_or_gate():
    block=ReadWriteConditionedMamba(4,d_state=2,expand=1,backend='torch')
    tokens=torch.randn(1,5,4)
    condition=torch.randn_like(tokens)
    first=block.scan_inputs(tokens,condition)
    second=block.scan_inputs(tokens,condition+torch.randn_like(condition))
    for i in (0,1,2):
        torch.testing.assert_close(first[i],second[i],rtol=0,atol=0)
    assert not torch.allclose(first[3],second[3])
    assert not torch.allclose(first[4],second[4])
    assert not hasattr(block,'x_proj')
    with pytest.raises(ValueError,match='identical'):
        block(tokens,condition[:,:3])


@pytest.mark.parametrize('size',[11,22])
def test_native_grids_multiscale_gradients_and_residual(size):
    block=MultiscaleReadWriteB(d_state=2,expand=1,backend='torch')
    assert [layer.kernel_size[0] for layer in block.scales]==[1,3,5,7]
    assert all(layer.groups==32 for layer in block.scales)
    image=torch.randn(1,32,size,size)
    output=block(image)
    assert output.shape==image.shape and block.mamba.last_backend=='torch'
    output.square().mean().backward()
    for name,p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name
    with torch.no_grad():
        block.output.weight.zero_();block.output.bias.zero_()
        torch.testing.assert_close(block(image),image,rtol=0,atol=0)
        block.residual=False
        assert torch.count_nonzero(block(image))==0


def test_fused_backend_rejects_cpu_instead_of_silent_fallback():
    block=ReadWriteConditionedMamba(4,backend='fused')
    with pytest.raises(RuntimeError,match='requires CUDA'):
        block(torch.randn(1,3,4),torch.randn(1,3,4))


def test_three_head_integration_with_original_a_c():
    model=PolypPVTSlots(FakeBaseline(),a=True,a_f2=True,b=True,c=True,cfm=False,
                       deep_supervision=True,b_variant='multiscale_readwrite',b_residual=True,
                       d_state=2,expand=1,backend='torch')
    assert type(model.b3) is MultiscaleReadWriteB and type(model.b4) is MultiscaleReadWriteB
    assert model.b3.mamba is not model.b4.mamba
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


def test_config_changes_only_b_and_output():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_readwrite_b.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]}=={'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)}=={'b_variant','b_residual'}
    assert cfg['augmentation']=='none' and cfg['aux_weights']==[.5,.5]


@pytest.mark.parametrize('options',[
    dict(b=False,b_variant='multiscale_readwrite'),
    dict(b=True,b_variant='multiscale_readwrite',b_scan='axial'),
    dict(a=True,a_f2=True,b=True,c=True,cfm=False,deep_supervision=True,
         ablation='wo_b',b_variant='multiscale_readwrite'),
])
def test_invalid_combinations(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)
