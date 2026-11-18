import json

import pytest
import torch
from test_data_metrics import make_manifest
from test_workflow import TinySegmenter
from torch.nn import functional as F

from medsam_research import engine
from medsam_research.backbone_controls import (
    BackboneControl,
    MultiScaleHead,
    evaluate_backbone_control,
    train_backbone_control,
)
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig


class PyramidTiny(TinySegmenter):
    def forward_with_pyramid(self, images):
        return self(images), [images, F.avg_pool2d(images, 2), F.avg_pool2d(images, 4)]


@pytest.mark.parametrize('mode', ['decoder', 'deep', 'pooled', 'multiscale'])
def test_initial_identity_and_gradient_isolation(mode):
    torch.manual_seed(4)
    baseline = PyramidTiny()
    model = BackboneControl(baseline, mode, [3, 3, 3], width=4)
    images = torch.rand(1, 3, 32, 32)
    expected = baseline(images).detach()
    model.train()
    output = model(images)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.square().mean().backward()
    if mode == 'decoder':
        assert baseline.conv.weight.grad.abs().sum() > 0
    else:
        assert all(p.grad is None for p in baseline.parameters())
        assert not baseline.training
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())


def test_fine_feature_changes_can_affect_multiscale_but_are_removed_by_pooling():
    # Exactly zero-mean high-frequency feature; pooled control removes this information.
    head = MultiScaleHead([1, 1, 1], width=1)
    for parameter in head.parameters():
        torch.nn.init.constant_(parameter, 1.)
    pattern = torch.tensor([[1., -1.], [-1., 1.]]).repeat(8, 8)[None, None]
    coarse = torch.zeros(1, 1, 4, 4)
    features = [pattern, torch.zeros(1, 1, 8, 8), coarse]
    baseline = torch.zeros(1, 1, 32, 32)
    pooled = [F.interpolate(F.adaptive_avg_pool2d(p, (4, 4)), p.shape[-2:], mode='bilinear',
                            align_corners=False) for p in features]
    assert not torch.allclose(head(features, baseline), head(pooled, baseline))


@pytest.mark.parametrize('mode', ['decoder', 'deep', 'pooled', 'multiscale'])
def test_train_save_restore_control(tmp_path, monkeypatch, mode):
    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    source = tmp_path/'source.pt'
    source.write_bytes(b'test source')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, batch_size=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path/'baseline')))
    monkeypatch.setattr(engine, 'build_model', lambda cfg, device: PyramidTiny().to(device))
    checkpoint = engine.train(cfg)/'best.pt'
    before = checkpoint.read_bytes()
    out = tmp_path/'control'
    scores = train_backbone_control(cfg, checkpoint, out, mode, epochs=2, width=4)
    assert 0 <= scores['dice'] <= 1
    assert before == checkpoint.read_bytes()
    summary = json.loads((out/'summary.json').read_text())
    assert (out/'paired_deltas.csv').exists()
    assert len((out/'history.jsonl').read_text().splitlines()) == 2
    restored = evaluate_backbone_control(cfg, checkpoint, out/'best_control.pt', val,
                                         tmp_path/'evaluate')
    assert restored == summary['modes']['adapted']
    assert len(list(out.glob('*_errors.png'))) == 2
    saved = torch.load(out/'best_control.pt', weights_only=True)
    saved['provenance']['baseline_checkpoint_sha256'] = 'wrong'
    torch.save(saved, out/'wrong.pt')
    with pytest.raises(ValueError, match='different baseline'):
        evaluate_backbone_control(cfg, checkpoint, out/'wrong.pt', val, tmp_path/'wrong')
