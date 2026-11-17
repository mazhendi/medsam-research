import json

import numpy as np
import pytest
import torch
from PIL import Image
from test_data_metrics import make_manifest
from test_workflow import TinySegmenter

from medsam_research import engine
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig
from medsam_research.diagnostics import (
    crop_boxes,
    diagnose,
    error_overlay,
    reread_logits,
    vessel_metrics,
)


def test_vessel_width_bands_and_empty_masks():
    y = np.zeros((40, 40), dtype=bool)
    y[3:30, 4] = True
    y[3:30, 20:31] = True
    p = y.copy()
    p[:, 4] = False
    scores = vessel_metrics(p, y, np.ones_like(y))
    assert scores['thin_skeleton_recall'] == 0
    assert scores['thick_skeleton_recall'] == 1
    assert scores['medium_skeleton_recall'] is None
    assert 0 < scores['cldice'] < 1
    assert vessel_metrics(y, y, np.ones_like(y))['cldice'] == 1
    empty = np.zeros_like(y)
    assert vessel_metrics(empty, empty, np.ones_like(y))['cldice'] == 1
    assert vessel_metrics(empty, y, np.ones_like(y))['cldice'] == 0
    assert vessel_metrics(y, empty, np.ones_like(y))['cldice'] == 0


@pytest.mark.parametrize('shape,grid,overlap', [((13, 19), 2, .25), ((1, 2), 3, 0),
                                              ((23, 31), 4, .5)])
def test_native_crop_assembly_has_full_coverage(shape, grid, overlap):
    class Constant(torch.nn.Module):
        image_size = 16

        def forward(self, image):
            return torch.full((len(image), 1, 16, 16), 2.0)

    image = Image.new('RGB', (shape[1], shape[0]))
    logits, boxes = reread_logits(Constant(), image, torch.device('cpu'), grid, overlap)
    torch.testing.assert_close(logits, torch.full(shape, 2.0))
    assert boxes == crop_boxes(image.width, image.height, grid, overlap)
    coverage = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        coverage[y0:y1, x0:x1] = True
    assert coverage.all()


def test_error_overlay_colors_and_fov():
    image = Image.new('RGB', (3, 1))
    overlay = np.array(error_overlay(image, [[0, 1, 1]], [[1, 0, 1]], [[1, 1, 0]]))
    assert overlay[0, 0, 0] > 0 and overlay[0, 0, 2] == 0
    assert overlay[0, 1, 2] > 0 and overlay[0, 1, 0] == 0
    assert not overlay[0, 2].any()


def test_diagnose_reuses_checkpoint_and_restricts_to_validation(tmp_path, monkeypatch):
    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    source = tmp_path / 'source.pt'
    source.write_bytes(b'test-only pretrained identity')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path / 'run')))
    monkeypatch.setattr(engine, 'build_model', lambda cfg, device: TinySegmenter().to(device))
    run = engine.train(cfg)
    out = tmp_path / 'diagnosis'
    summary = diagnose(cfg, run / 'best.pt', val, out)
    assert summary['global']['model_calls'] == 1
    assert summary['reread']['model_calls'] == 5
    assert len(list(out.glob('*_errors.png'))) == 2
    assert json.loads((out / 'summary.json').read_text())['samples'] == 1
    with pytest.raises(FileExistsError):
        diagnose(cfg, run / 'best.pt', val, out)
    with pytest.raises(ValueError, match='validation manifest'):
        diagnose(cfg, run / 'best.pt', train, tmp_path / 'wrong')
