import csv
import json

import numpy as np
import pytest
import torch
from test_backbone_controls import PyramidTiny
from test_data_metrics import make_manifest
from torch import nn
from torch.nn import functional as F

from medsam_research import engine, text_training
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig
from medsam_research.text_evidence import read_prompt_map, resolve_prompts
from medsam_research.text_training import TextSystem, predict_text_model, train_text_model
from medsam_research.text_verification import (
    DirectionalRepair,
    TextLocalizer,
    complementarity,
    direction_loss,
    intervene_prior,
    repair_objective,
)


class FixtureEvidence(nn.Module):
    """Workflow fixture only; NOT BiomedCLIP or evidence of image/text accuracy."""

    text_dim = 8
    identity = {'test_fixture': 'not-biomedclip'}

    def __init__(self, directory, device):
        super().__init__()
        self.register_buffer('anchor', torch.zeros(1, device=device))

    def forward(self, images, prompts):
        text = torch.tensor([[((sum(p.encode())+j) % 13)/13 for j in range(self.text_dim)]
                             for p in prompts], device=images.device)
        similarity = F.adaptive_avg_pool2d(images.mean(1, keepdim=True), 4)*text[:, :1, None, None]
        return similarity, text


def tensors():
    torch.manual_seed(1)
    pyramid = [torch.rand(1, 3, 16, 16), torch.rand(1, 3, 8, 8), torch.rand(1, 3, 4, 4)]
    visual = torch.randn(1, 1, 32, 32)
    prior = torch.randn_like(visual)
    target = torch.zeros_like(prior)
    target[:, :, 5:20, 8:22] = 1
    return pyramid, visual, prior, target


def test_localizer_and_repair_identity_with_disjoint_domains():
    pyramid, visual, prior, target = tensors()
    localizer = TextLocalizer([3]*3, text_dim=8, width=4)
    result = localizer(pyramid, visual, torch.rand(1, 1, 4, 4), torch.rand(1, 8))
    torch.testing.assert_close(result, visual, atol=0, rtol=0)
    head = DirectionalRepair([3]*3, width=4)
    fields = head(pyramid, visual, prior)
    torch.testing.assert_close(fields['logits'], prior, atol=0, rtol=0)
    loss = direction_loss(fields, target, torch.ones_like(target))
    loss.backward()
    assert head.output.weight.grad.abs().sum() > 0
    # Empty foreground operation domain is legal, finite and differentiable.
    head.zero_grad(set_to_none=True)
    fields = head(pyramid, visual, torch.full_like(prior, -4))
    direction_loss(fields, target, torch.ones_like(target)).backward()
    assert torch.isfinite(head.output.weight.grad).all()


@pytest.mark.parametrize('mode', ['full', 'no-intervention', 'no-direction', 'residual', 'balanced-residual'])
def test_repair_loss_is_finite_and_frozen_inputs_receive_no_grad(mode):
    pyramid, visual, prior, target = tensors()
    head = DirectionalRepair([3]*3, width=4, mode=mode)
    loss = repair_objective(head, pyramid, visual, prior, target, torch.ones_like(target), mode,
                            generator=torch.Generator().manual_seed(42))
    loss.backward()
    assert torch.isfinite(loss) and head.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in pyramid)


def test_intervention_is_deterministic_label_free_and_changes_domains():
    prior = torch.zeros(2, 1, 32, 32)
    first = intervene_prior(prior, torch.Generator().manual_seed(3))
    second = intervene_prior(prior, torch.Generator().manual_seed(3))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert (first != prior).any() and not prior.any()


def test_complementarity_counts_and_prompt_validation(tmp_path):
    target = np.array([[1, 1, 0, 0]], dtype=bool)
    text = np.array([[0, 1, 1, 0]], dtype=bool)
    visual = target.copy()
    scores = complementarity(text, visual, target, np.ones_like(target))
    assert scores['text_fn_pixels'] == scores['text_fp_pixels'] == 1
    assert scores['visual_recovery_fraction'] == scores['visual_rejection_fraction'] == 1
    mapping = tmp_path/'prompts.json'
    mapping.write_text(json.dumps({'aa': 'blood vessels'}))
    assert resolve_prompts(['aa', 'bb'], 'polyp', read_prompt_map(mapping)) == ['blood vessels', 'polyp']
    mapping.write_text(json.dumps({'aa': 3}))
    with pytest.raises(ValueError):
        read_prompt_map(mapping)


def test_training_stages_evaluation_and_label_free_inference(tmp_path, monkeypatch):
    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    source = tmp_path/'source.pt'
    source.write_bytes(b'test source')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, batch_size=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path/'baseline')))
    monkeypatch.setattr(engine, 'build_model', lambda cfg, device: PyramidTiny().to(device))
    monkeypatch.setattr(text_training, 'FrozenBiomedCLIP', FixtureEvidence)
    checkpoint = engine.train(cfg)/'best.pt'
    original = checkpoint.read_bytes()
    local_out, repair_out = tmp_path/'local', tmp_path/'repair'
    train_text_model(cfg, checkpoint, 'fixture', local_out, epochs=2, width=4, task_only=True)
    initial = json.loads((local_out/'initial_validation.json').read_text())
    assert initial['visual'] == initial['text'] == initial['text_null']
    local_path = local_out/'last_localizer.pt'
    local_original = local_path.read_bytes()
    train_text_model(cfg, checkpoint, 'fixture', repair_out, stage='repair',
                     localizer_path=local_path, epochs=2)
    initial = json.loads((repair_out/'initial_validation.json').read_text())
    assert initial['text'] == initial['repaired']
    assert checkpoint.read_bytes() == original and local_path.read_bytes() == local_original
    assert (repair_out/'complementarity.csv').exists()
    assert (repair_out/'repair_transitions.csv').exists()
    repair_path = repair_out/'last_repair.pt'
    saved = torch.load(repair_path, weights_only=True)
    assert saved['state']['output.weight'].abs().sum() > 0
    scores = text_training.evaluate_text_model(cfg, checkpoint, 'fixture', local_path, val,
                                               tmp_path/'evaluation', repair_path)
    assert 0 <= scores['repaired']['dice'] <= 1
    inference = tmp_path/'inference.csv'
    with inference.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'image', 'mask', 'fov'])
        writer.writerow(['cc', 'cc.png', 'missing_mask.png', 'missing_fov.png'])
    output = predict_text_model(cfg, checkpoint, 'fixture', local_path, inference,
                                tmp_path/'inference', repair_path)
    assert (output/'000000.png').exists()
    saved['provenance']['localizer_sha256'] = 'wrong'
    torch.save(saved, repair_out/'wrong.pt')
    with pytest.raises(ValueError, match='another localizer'):
        predict_text_model(cfg, checkpoint, 'fixture', local_path, inference,
                           tmp_path/'wrong', repair_out/'wrong.pt')


def test_stage_two_freezes_both_upstream_branches():
    baseline = PyramidTiny()
    system = TextSystem(baseline, FixtureEvidence('', 'cpu'), [3]*3, width=4,
                         repair_mode='full').train()
    result = system(torch.rand(1, 3, 32, 32), ['vessels'])
    result['repaired'].square().mean().backward()
    assert all(p.grad is None for p in system.baseline.parameters())
    assert all(p.grad is None for p in system.localizer.parameters())
    assert all(p.grad is None for p in system.evidence.parameters())
    assert system.repair.output.weight.grad.abs().sum() > 0


def test_nonzero_localizer_uses_spatial_and_text_inputs():
    pyramid, visual, _, _ = tensors()
    head = TextLocalizer([3]*3, text_dim=8, width=4)
    nn.init.normal_(head.output.weight, std=.1)
    response = torch.rand(1, 1, 4, 4, requires_grad=True)
    text = torch.rand(1, 8, requires_grad=True)
    result = head(pyramid, visual, response, text)
    null = head(pyramid, visual, torch.zeros_like(response), torch.zeros_like(text))
    assert not torch.allclose(result, null)
    result.square().mean().backward()
    assert response.grad.abs().sum() > 0 and text.grad.abs().sum() > 0


def test_directional_composition_applies_correct_sign_in_each_domain():
    pyramid, visual, prior, _ = tensors()
    head = DirectionalRepair([3]*3, width=4)
    with torch.no_grad():
        head.output.bias.copy_(torch.tensor([2., 3.]))
    result = head(pyramid, visual, prior)
    expected = torch.where(prior >= 0, prior-3, prior+2)
    torch.testing.assert_close(result['logits'], expected)


def test_bundle_hash_and_path_validation(tmp_path):
    from medsam_research.data import file_hash
    from medsam_research.text_evidence import bundle_identity

    files = ['open_clip_config.json', 'open_clip_pytorch_model.bin', 'tokenizer/config.json']
    for name in files:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    receipt = {'files': {name: file_hash(tmp_path/name) for name in files}}
    (tmp_path/'bundle.json').write_text(json.dumps(receipt))
    assert bundle_identity(tmp_path) == receipt
    (tmp_path/files[0]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='bundle changed'):
        bundle_identity(tmp_path)
    receipt['files'] = {'../outside': 'fake'}
    (tmp_path/'bundle.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='bundle changed'):
        bundle_identity(tmp_path)


def test_direction_loss_is_explicitly_equivalent_to_domain_balanced_final_bce():
    pyramid, visual, prior, target = tensors()
    head = DirectionalRepair([3]*3, width=4)
    nn.init.normal_(head.output.weight, std=.1)
    result = head(pyramid, visual, prior)
    valid = torch.ones_like(target)
    errors = F.binary_cross_entropy_with_logits(result['logits'], target, reduction='none')
    domain = result['domain']
    expected = (errors*(~domain)).sum()/(~domain).sum().clamp_min(1)
    expected = expected+(errors*domain).sum()/domain.sum().clamp_min(1)
    torch.testing.assert_close(direction_loss(result, target, valid), expected)


def test_per_image_training_requires_coverage_and_carries_prompt_policy(tmp_path, monkeypatch):
    from medsam_research.data import file_hash

    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    source = tmp_path/'source.pt'
    source.write_bytes(b'test source')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, batch_size=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path/'baseline')))
    monkeypatch.setattr(engine, 'build_model', lambda cfg, device: PyramidTiny().to(device))
    monkeypatch.setattr(text_training, 'FrozenBiomedCLIP', FixtureEvidence)
    checkpoint = engine.train(cfg)/'best.pt'
    with pytest.raises(ValueError, match='prompt-map'):
        train_text_model(cfg, checkpoint, 'fixture', tmp_path/'no_map', epochs=1)
    path = tmp_path/'captions.json'
    path.write_text(json.dumps({'aa': 'dark image'}))
    with pytest.raises(ValueError, match='Missing per-image'):
        train_text_model(cfg, checkpoint, 'fixture', tmp_path/'missing', prompt_map=path, epochs=1)
    path.write_text(json.dumps(dict.fromkeys(['aa', 'bb', 'cc'], 'same caption')))
    with pytest.raises(ValueError, match='identical'):
        train_text_model(cfg, checkpoint, 'fixture', tmp_path/'same', prompt_map=path, epochs=1)
    path.write_text(json.dumps({'aa': 'dark image', 'bb': 'bright image', 'cc': 'low contrast image'}))
    receipt = {'format_version': 1, 'task': 'retinal blood vessels',
               'image_sha256': {sid: file_hash(tmp_path/f'{sid}.png') for sid in ['aa', 'bb', 'cc']}}
    (tmp_path/'captions.metadata.json').write_text(json.dumps(receipt))
    out = tmp_path/'per_image'
    train_text_model(cfg, checkpoint, 'fixture', out, prompt_map=path, epochs=1, width=4)
    artifact = out/'last_localizer.pt'
    saved = torch.load(artifact, weights_only=True)
    assert saved['provenance']['prompt_policy'] == 'per-image'
    assert saved['provenance']['prompt_audit']['train']['image_hashes_checked'] == 2
    with (out/'images.csv').open() as stream:
        assert list(csv.DictReader(stream))[0]['prompt'] == 'low contrast image'
    inference = tmp_path/'inference.csv'
    inference.write_text('id,image,mask\nnew,cc.png,missing.png\n')
    with pytest.raises(ValueError, match='Missing per-image'):
        predict_text_model(cfg, checkpoint, 'fixture', artifact, inference, tmp_path/'bad_predict')
    override = tmp_path/'new_prompts.json'
    override.write_text(json.dumps({'new': 'new image descriptor'}))
    result = predict_text_model(cfg, checkpoint, 'fixture', artifact, inference,
                                tmp_path/'new_prediction', prompt_map=override)
    with (result/'predictions.csv').open() as stream:
        assert list(csv.DictReader(stream))[0]['prompt'] == 'new image descriptor'
