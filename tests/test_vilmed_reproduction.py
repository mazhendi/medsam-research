"""Harness checks; component gradients do not imply reproduced paper scores."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_data_metrics import make_manifest

from medsam_research.data import file_hash, read_manifest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'/'simple_vilmedsam'))
spec = importlib.util.spec_from_file_location('vilmed_harness', ROOT/'experiments'/'simple_vilmedsam'/'run.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def test_squared_dice_bce_has_useful_gradients():
    target = torch.tensor([[[[1., 0.], [0., 1.]]]])
    logits = torch.zeros_like(target, requires_grad=True)
    loss = harness.dice_bce(logits, target)
    loss.backward()
    assert (logits.grad[target.bool()] < 0).all()
    assert (logits.grad[~target.bool()] > 0).all()
    assert harness.dice_bce((target*2-1)*10, target) < loss.detach()/100


def test_map_image_binding_and_rectangular_geometry(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path, 'samples.csv', ['aa'])
    record = read_manifest(manifest)[0]
    values = np.zeros((12, 24), dtype=np.uint8)
    values[3:9, 8:16] = 255
    Image.fromarray(np.repeat(values[:, :, None], 3, axis=2)).save(record.image)
    Image.fromarray(values).save(record.mask)
    root = tmp_path/'maps'
    root.mkdir()
    Image.fromarray(values).save(root/'map.png')
    entry = {'file': 'map.png', 'image_sha256': file_hash(record.image),
             'map_sha256': file_hash(root/'map.png')}
    (root/'maps.json').write_text(json.dumps({'status': 'complete', 'samples': {'aa': entry}}))
    x, y, a, native, shape = harness.padded_sample(record, root, size=48)
    assert shape == (24, 48) and native.shape == (12, 24)
    assert y[:, :, 24:].sum() == 0 and a[:, :, 24:].sum() == 0
    torch.testing.assert_close(y*255, a)
    monkeypatch.setattr(harness, 'forward', lambda sam, decoder, image, attribution:
                        image[:, :1]/25.5-5)
    rows = harness.evaluate(torch.nn.Identity(), torch.nn.Identity(), [record], root, 'cpu')
    assert rows[0]['dice'] > .99 and rows[0]['recall'] > .99
    values[0, 0] = 255
    Image.fromarray(values).save(root/'map.png')
    with pytest.raises(ValueError, match='identity'):
        harness.padded_sample(record, root, size=48)


@pytest.mark.integration
def test_author_lora_initial_identity_gradient_and_reload():
    source = ROOT/'vendor'/'Simple-ViLMedSAM'
    if not (source/'reproduction_patch.json').exists():
        pytest.skip('Run experiments/simple_vilmedsam/prepare.py first')
    _, LoRA, _, _, _ = harness.author_modules(source)
    from segment_anything.modeling.image_encoder import ImageEncoderViT
    encoder = ImageEncoderViT(img_size=32, patch_size=8, embed_dim=32, depth=2,
                              num_heads=4, out_chans=16, global_attn_indexes=[1], window_size=2)
    shell = torch.nn.Module()
    shell.image_encoder = encoder
    shell.requires_grad_(False)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original = encoder(x).clone()
    lora = LoRA(shell, 4)
    torch.testing.assert_close(encoder(x), original)
    saved = harness.adaptation_state(lora)
    trainable = [p for p in lora.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.01)
    # A nonconstant spatial probe avoids LayerNorm's nearly constant squared norm.
    (encoder(x)*torch.randn_like(original)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    optimizer.step()
    assert any(not torch.equal(value, saved[name]) for name, value in harness.adaptation_state(lora).items())
    harness.restore_adapter(lora, saved)
    torch.testing.assert_close(encoder(x), original)
    assert not encoder.patch_embed.proj.weight.requires_grad
