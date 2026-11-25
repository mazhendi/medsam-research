from pathlib import Path

import pytest
import torch

from medsam_research.abc_decoder import SpatialContextA, ChannelSpatialC
from medsam_research.cross_scale_b1 import CrossScaleInteraction, CrossScaleMambaB1
from medsam_research.polyp_pvt_slots import PolypPVTSlots, slots_loss, validate_slot_options
from medsam_research.polyp_pvt_workflow import load_pvt_config
from test_polyp_pvt_slots import FakeBaseline


def test_scale_attention_excludes_self_and_preserves_own_concat():
    block = CrossScaleInteraction(2)
    # Uniform cross-scale weights, identity values, select own channel 0 and
    # message channel 0 respectively in the two output channels.
    with torch.no_grad():
        for q,k,v,fuse in zip(block.query,block.key,block.value,block.fuse):
            q.weight.zero_(); q.bias.zero_()
            k.weight.zero_(); k.bias.zero_()
            v.weight.zero_(); v.bias.zero_()
            v.weight[0,0,0,0] = v.weight[1,1,0,0] = 1
            fuse.weight.zero_(); fuse.bias.zero_()
            fuse.weight[0,0,0,0] = fuse.weight[1,2,0,0] = 1
    features = [torch.full((1,2,2,3),float(n)) for n in (1,3,9)]
    outputs,weights = block(features,return_weights=True)
    for i in range(3):
        others = [j for j in range(3) if j != i]
        expected = torch.cat((features[i][:,:1],
                              (features[others[0]][:,:1]+features[others[1]][:,:1])/2),1)
        torch.testing.assert_close(outputs[i],expected)
        torch.testing.assert_close(weights[i],torch.full((1,2,2,3),.5))


def test_scale_interaction_does_not_mix_spatial_locations():
    block = CrossScaleInteraction(4).eval()
    features = [torch.randn(1,4,3,4) for _ in range(3)]
    before = block(features)
    modified = [x.clone() for x in features]
    modified[1][:,:,1,2] += 10
    after = block(modified)
    mask = torch.ones(3,4,dtype=torch.bool); mask[1,2] = False
    for old,new in zip(before,after):
        torch.testing.assert_close(old[:,:,mask],new[:,:,mask],rtol=0,atol=0)


@pytest.mark.parametrize('size',[11,22])
def test_b1_native_grid_and_all_gradients(size):
    block = CrossScaleMambaB1(d_state=2,expand=1,backend='torch')
    assert block.project.out_channels == 48
    assert [x.kernel_size for x in block.local] == [(3,3),(5,5),(7,7)]
    output = block(torch.randn(1,32,size,size))
    assert output.shape == (1,32,size,size)
    output.square().mean().backward()
    for name,p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum() > 0,name
    # Zero final projection must yield zero: no outer residual path.
    with torch.no_grad():
        block.fuse.weight.zero_(); block.fuse.bias.zero_()
        assert torch.count_nonzero(block(torch.randn(1,32,size,size))) == 0


def test_three_head_b1_replaces_both_blocks_preserves_a_c_and_supervision():
    model = PolypPVTSlots(FakeBaseline(),a=True,a_f2=True,b=True,c=True,
                         cfm=False,deep_supervision=True,b_variant='cross_scale_b1',
                         d_state=2,expand=1,backend='torch')
    assert isinstance(model.b3,CrossScaleMambaB1) and isinstance(model.b4,CrossScaleMambaB1)
    assert model.b3.project.weight is not model.b4.project.weight
    assert type(model.cim) is SpatialContextA and type(model.a2) is SpatialContextA
    assert type(model.SAM) is ChannelSpatialC
    image = torch.randn(2,3,64,64)
    seen=[]
    for head in (model.aux3,model.aux4):
        head.register_forward_pre_hook(lambda _,args:seen.append(args[0].shape[-2:]))
    outputs = model(image)
    assert set(outputs) == {'final','aux3','aux4'} and seen == [(4,4),(2,2)]
    slots_loss(outputs,torch.rand(2,1,64,64)).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    model.eval()
    with torch.no_grad():
        assert set(model(image)) == {'final'}
    assert len(seen) == 2


def test_b1_config_changes_only_variant_and_output():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/'configs/polyp_pvt_slots_three_head.yaml')
    cfg=load_pvt_config(root/'configs/polyp_pvt_slots_three_head_b1.yaml')
    assert {k for k in cfg if cfg[k]!=base[k]} == {'slots','output'}
    assert {k for k in cfg['slots'] if cfg['slots'][k]!=base['slots'].get(k)} == {'b_variant'}
    assert cfg['augmentation'] == 'none' and cfg['aux_weights'] == [.5,.5]


@pytest.mark.parametrize('options',[
    dict(b=True,b_variant='invalid'), dict(b=False,b_variant='cross_scale_b1'),
    dict(b=True,b_variant='cross_scale_b1',b_residual=True),
    dict(b=True,b_variant='cross_scale_b1',b_scan='axial'),
    dict(a=True,a_f2=True,b=True,c=True,cfm=False,deep_supervision=True,
         ablation='wo_b',b_variant='cross_scale_b1'),
])
def test_invalid_b1_combinations_rejected(options):
    with pytest.raises(ValueError):
        validate_slot_options(options)
