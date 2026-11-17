import json

import numpy as np
import torch
from test_data_metrics import make_manifest
from test_workflow import TinySegmenter

from medsam_research import engine
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig
from medsam_research.correction import CorrectionHead, center_boxes, read_box, roi, train_correction


def test_centers_partition_and_read_actions():
    coverage = np.zeros((13, 19), dtype=int)
    for box in center_boxes(19, 13, 3):
        x0, y0, x1, y1 = box
        coverage[y0:y1, x0:x1] += 1
        assert read_box(box, 19, 13, 0) == box
        cx0, cy0, cx1, cy1 = read_box(box, 19, 13, 1)
        assert 0 <= cx0 <= x0 < x1 <= cx1 <= 19
        assert 0 <= cy0 <= y0 < y1 <= cy1 <= 13
    assert (coverage == 1).all()


def test_roi_coordinates_and_identity_head_gradients():
    tensor = torch.arange(24).float().reshape(1, 1, 4, 6)
    # Exact pixel-aligned 2x2 region from a rectangular feature map.
    torch.testing.assert_close(roi(tensor, (2, 1, 4, 3), (0, 0, 6, 4), 2),
                               tensor[:, :, 1:3, 2:4])
    head = CorrectionHead(3, hidden=4)
    g, local = torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)
    base, patch = torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8)
    result = head(g, local, base, patch, 0)
    torch.testing.assert_close(result, base, atol=0, rtol=0)
    result.square().mean().backward()
    assert head.body[-1].weight.grad.abs().sum() > 0


def test_correction_training_with_existing_baseline(tmp_path, monkeypatch):
    class FeatureTiny(TinySegmenter):
        def forward_with_features(self, images):
            return self(images), images

    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    source = tmp_path / 'source.pt'
    source.write_bytes(b'test-only pretrained identity')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path / 'baseline')))
    monkeypatch.setattr(engine, 'build_model', lambda cfg, device: FeatureTiny().to(device))
    run = engine.train(cfg)
    baseline_before = (run / 'best.pt').read_bytes()
    out = tmp_path / 'correction'
    train_correction(cfg, run / 'best.pt', out, epochs=2, size=16)
    initial = json.loads((out / 'initial_validation.json').read_text())
    assert initial['global'] == initial['detail'] == initial['context']
    assert (run / 'best.pt').read_bytes() == baseline_before
    saved = torch.load(out / 'last_correction.pt', weights_only=True)
    assert saved['head']['body.4.weight'].abs().sum() > 0
    assert len(list(out.glob('*_errors.png'))) == 3
    assert len((out / 'history.jsonl').read_text().splitlines()) == 2
