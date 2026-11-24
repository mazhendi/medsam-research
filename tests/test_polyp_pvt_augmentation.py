import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from medsam_research.polyp_pvt_augmentation import augment_pair, augmentation_policy
from medsam_research.polyp_pvt_workflow import PolypPVTDataset, load_pvt_config


class Decisions:
    def __init__(self, values):
        self.values = iter(values)
    def random(self):
        return next(self.values)
    def choice(self, values):
        return values[0]
    def shuffle(self, values):
        pass
    def uniform(self, low, high):
        return high


def pair():
    mask = np.zeros((8,8),dtype=np.uint8)
    mask[1:3,2:5] = 255
    mask[6,1] = 127
    return Image.fromarray(np.repeat(mask[...,None],3,axis=2)), Image.fromarray(mask)


@pytest.mark.parametrize('decisions,transform', [([0,1,1,1],lambda a:np.fliplr(a)),
                                                ([1,0,1,1],lambda a:np.flipud(a)),
                                                ([1,1,0,1],lambda a:np.rot90(a))])
def test_geometry_is_exact_and_paired_without_mask_interpolation(decisions,transform):
    image,mask=pair()
    a,b=augment_pair(image,mask,'polyp_light',Decisions(decisions))
    np.testing.assert_array_equal(np.asarray(b),transform(np.asarray(mask)))
    np.testing.assert_array_equal(np.asarray(a)[...,0],np.asarray(b))
    assert sorted(np.unique(b)) == [0,127,255]


def test_appearance_changes_only_image_and_preserves_soft_mask():
    image=Image.fromarray(np.full((8,8,3),100,dtype=np.uint8))
    _,mask=pair()
    a,b=augment_pair(image,mask,'polyp_light',Decisions([1,1,1,0]))
    assert not np.array_equal(a,image)
    np.testing.assert_array_equal(b,mask)


def test_none_preserves_identity_and_does_not_consume_rng():
    image,mask=pair()
    before=random.getstate()
    a,b=augment_pair(image,mask,'none')
    assert a is image and b is mask
    assert random.getstate() == before


def test_seeded_augmentation_is_reproducible():
    image,mask=pair()
    one=augment_pair(image,mask,'polyp_light',random.Random(42))
    two=augment_pair(image,mask,'polyp_light',random.Random(42))
    for a,b in zip(one,two):
        np.testing.assert_array_equal(a,b)


def test_dataset_legacy_none_parity_and_augmented_soft_targets(tmp_path):
    from test_polyp_pvt import write_data
    from medsam_research.polyp_pvt_workflow import image_tensor
    records=write_data(tmp_path)
    old_x,old_y=PolypPVTDataset(records,32)[0]
    no_x,no_y=PolypPVTDataset(records,32,'none')[0]
    torch.testing.assert_close(old_x,no_x,rtol=0,atol=0)
    torch.testing.assert_close(old_y,no_y,rtol=0,atol=0)
    with Image.open(records[0].image) as image:
        torch.testing.assert_close(old_x,image_tensor(image.convert('RGB').resize((32,32),Image.Resampling.BILINEAR)),rtol=0,atol=0)
    x,y=PolypPVTDataset(records,32,'polyp_light')[0]
    assert x.shape == (3,32,32) and y.shape == (1,32,32)
    assert torch.isfinite(x).all() and 0 <= y.min() <= y.max() <= 1
    assert ((y>0)&(y<1)).any()


def test_augmented_configs_change_only_augmentation_and_output():
    root=Path(__file__).resolve().parents[1]/'configs'
    names=['polyp_pvt_1450','polyp_pvt_slots_three_head','polyp_pvt_slots_three_head_no_aux']
    for name in names:
        old=load_pvt_config(root/f'{name}.yaml')
        new=load_pvt_config(root/f'{name}_aug.yaml')
        assert {key for key in old if old[key]!=new[key]} == {'augmentation','output'}
        assert new['augmentation'] == 'polyp_light'
        assert new['output'] == old['output']+'_aug'


def test_augmented_training_records_policy_and_preserves_update_budget(tmp_path,monkeypatch):
    import yaml
    from test_polyp_pvt import write_config,write_data,TinyTwoHead
    from medsam_research import polyp_pvt_workflow as workflow
    path,cfg=write_config(tmp_path)
    cfg['augmentation']='polyp_light'
    path.write_text(yaml.safe_dump(cfg))
    write_data(tmp_path)
    (tmp_path/'encoder.pth').write_bytes(b'probe')
    monkeypatch.setattr(workflow,'build_polyp_pvt',lambda *args:TinyTwoHead())
    report=workflow.train_pvt(path)
    saved=torch.load(report['checkpoint'],weights_only=True)
    assert report['steps'] == 3
    assert saved['provenance']['augmentation_policy'] == augmentation_policy('polyp_light')
    assert saved['provenance']['training_sizes'] == [256,352,448]
    with pytest.raises(FileExistsError):
        workflow.train_pvt(path)
