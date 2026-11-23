import copy
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from medsam_research.polyp_pvt import build_polyp_pvt
from medsam_research.polyp_pvt_slots import PolypPVTSlots
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.sam2_unet_workflow import structure_loss


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.ModuleList(nn.Conv2d(3, n, 1) for n in (64,128,320,512))

    def forward(self, x):
        return [m(F.avg_pool2d(x, k)) for m,k in zip(self.project, (4,8,16,32))]


class FakeCFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(96,32,1)

    def forward(self, deep, middle, fine):
        return self.fuse(torch.cat((F.interpolate(deep, fine.shape[-2:]),
                                    F.interpolate(middle, fine.shape[-2:]),fine),1))


class FakeSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(64,32,1)

    def forward(self, main, detail):
        return self.fuse(torch.cat((main,detail),1))


class FakeBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeEncoder()
        self.Translayer2_0 = nn.Conv2d(64,32,1)
        self.Translayer2_1 = nn.Conv2d(128,32,1)
        self.Translayer3_1 = nn.Conv2d(320,32,1)
        self.Translayer4_1 = nn.Conv2d(512,32,1)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(64,64,1),nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(64,1,1),nn.Sigmoid())
        self.CFM, self.SAM = FakeCFM(), FakeSAM()
        self.down05 = nn.Upsample(scale_factor=.5,mode="bilinear",align_corners=True)
        self.out_CFM, self.out_SAM = nn.Conv2d(32,1,1), nn.Conv2d(32,1,1)

    def forward(self, x):
        f1,f2,f3,f4 = self.backbone(x)
        fine = self.ca(f1)*f1
        fine = self.sa(fine)*fine
        main = self.CFM(self.Translayer4_1(f4),self.Translayer3_1(f3),self.Translayer2_1(f2))
        detail = self.down05(self.Translayer2_0(fine))
        refined = self.SAM(main,detail)
        return (F.interpolate(self.out_CFM(main),scale_factor=8,mode="bilinear",align_corners=False),
                F.interpolate(self.out_SAM(refined),scale_factor=8,mode="bilinear",align_corners=False))


def test_all_off_reproduces_original_graph_and_checkpoint_keys():
    baseline = FakeBaseline().eval()
    model = PolypPVTSlots(copy.deepcopy(baseline)).eval()
    assert set(model.state_dict()) == set(baseline.state_dict())
    model.load_state_dict(baseline.state_dict(),strict=True)
    x = torch.randn(2,3,64,64)
    for original,controlled in zip(baseline(x),model(x)):
        torch.testing.assert_close(original,controlled,rtol=0,atol=0)


@pytest.mark.parametrize("a,b,c", [(False,False,False),(True,False,False),
    (False,True,False),(False,False,True),(True,True,True)])
def test_variants_keep_cfm_head_grids_and_finite_gradients(a,b,c):
    baseline = FakeBaseline()
    model = PolypPVTSlots(baseline,a=a,b=b,c=c,d_state=2,expand=1,backend="torch")
    assert model.CFM is baseline.CFM
    assert model.out_CFM is baseline.out_CFM and model.out_SAM is baseline.out_SAM
    seen=[]
    for head in (model.out_CFM,model.out_SAM):
        head.register_forward_pre_hook(lambda _,args:seen.append(tuple(args[0].shape)))
    y = (torch.rand(2,1,64,64)>.7).float()
    predictions=model(torch.randn(2,3,64,64))
    assert seen == [(2,32,8,8)]*2
    assert all(v.shape == y.shape for v in predictions)
    sum(structure_loss(v,y) for v in predictions).backward()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
    if a:
        assert not model.cim.residual
        assert not hasattr(model,"ca")
    if b:
        assert not model.b3.residual and not model.b4.residual


def test_interface_configs_preserve_original_training_protocol():
    root=Path(__file__).resolve().parents[1]
    base=load_pvt_config(root/"configs/polyp_pvt_1450.yaml")
    outputs=set()
    for name in ("control","a_only","b_only","c_only","full"):
        cfg=load_pvt_config(root/f"configs/polyp_pvt_slots_{name}.yaml")
        for key in ("epochs","batch_size","scales","learning_rate","weight_decay","seed"):
            assert cfg[key] == base[key]
        outputs.add(cfg["output"])
    assert len(outputs) == 5


@pytest.mark.integration
def test_real_official_all_off_exact_parity():
    source=os.environ.get("POLYP_PVT_SOURCE")
    if not source:
        pytest.skip("Set POLYP_PVT_SOURCE to test pinned official model parity")
    baseline=build_polyp_pvt(source).eval()
    model=PolypPVTSlots(baseline).eval()
    assert set(model.state_dict()) == set(baseline.state_dict())
    with torch.inference_mode():
        for size in (256,352,448):
            x=torch.randn(1,3,size,size)
            for expected,actual in zip(baseline(x),model(x)):
                torch.testing.assert_close(expected,actual,rtol=0,atol=0)
