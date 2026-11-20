"""Mechanism, checkpoint and real SAM2 API tests, not performance evidence."""
import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_data_metrics import make_manifest
from torch import nn
from torch.nn import functional as F

from medsam_research import engine
from medsam_research import region_shape_training as workflow
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig
from medsam_research.region_shape import (
    RegionShapeConfig,
    RegionShapeHead,
    RegionShapeSAM2,
    coarse_targets,
    load_mechanism_config,
    region_shape_loss,
    weighted_hypotheses,
)


class FixturePromptEncoder(nn.Module):
    embed_dim = 16

    def __init__(self):
        super().__init__()
        self.no_mask_embed = nn.Embedding(1, 16)
        self.register_buffer('pe', torch.zeros(1, 16, 4, 4))

    def get_dense_pe(self):
        return self.pe


class FixtureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s0 = nn.Conv2d(16, 4, 1)
        self.conv_s1 = nn.Conv2d(16, 8, 1)
        self.output = nn.Conv2d(16, 1, 1)
        self.forced_output = None

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, repeat_image, high_res_features):
        if self.forced_output is not None:
            masks = image_embeddings.new_full((len(image_embeddings), 1, 16, 16), self.forced_output)
        else:
            x = image_embeddings+dense_prompt_embeddings+sparse_prompt_embeddings.mean(1)[:, :, None, None]
            masks = F.interpolate(self.output(x), (16, 16), mode='bilinear', align_corners=False)
            masks = masks+.1*high_res_features[0].mean(1, keepdim=True)
        return masks, None, None, None


class FixtureBaseline(nn.Module):
    """Small shape-compatible workflow fixture; not SAM or an accuracy result."""
    image_size = 32

    def __init__(self):
        super().__init__()
        self.sam = nn.Module()
        self.sam.image_encoder = nn.Conv2d(3, 16, 1)
        self.sam.sam_prompt_encoder = FixturePromptEncoder()
        self.sam.sam_mask_decoder = FixtureDecoder()
        self.sam.requires_grad_(False)
        self.sam.sam_mask_decoder.requires_grad_(True)
        self.task_tokens = nn.Parameter(torch.randn(1, 4, 16)*.02)
        self.override_initial = None

    def forward_with_pyramid(self, x):
        image = self.sam.image_encoder(x)
        decoder = self.sam.sam_mask_decoder
        f0 = decoder.conv_s0(F.avg_pool2d(image, 2))
        f1 = decoder.conv_s1(F.avg_pool2d(image, 4))
        f2 = F.avg_pool2d(image, 8)
        dense = self.sam.sam_prompt_encoder.no_mask_embed.weight[0][None, :, None, None]
        masks, _, _, _ = decoder(f2, None, self.task_tokens.expand(len(x), -1, -1),
                                  dense.expand(len(x), -1, 4, 4), False, False, [f0, f1])
        logits = F.interpolate(masks, x.shape[-2:], mode='bilinear', align_corners=False)
        if self.override_initial is not None:
            logits = torch.full_like(logits, self.override_initial)
        return logits, [f0, f1, f2]

    def forward(self, x):
        return self.forward_with_pyramid(x)[0]


def settings(**kwargs):
    return RegionShapeConfig(width=8, heads=2, hypotheses=3, grid_size=8, rounds=2, **kwargs)


def test_weighted_hypotheses_moments_and_permutation():
    h = torch.tensor([[[[0., 1.]], [[1., 0.]]]])
    w = torch.tensor([[.25, .75]])
    mean, variance = weighted_hypotheses(h, w)
    torch.testing.assert_close(mean, torch.tensor([[[[.75, .25]]]]))
    torch.testing.assert_close(variance, torch.full_like(mean, .1875))
    a, b = weighted_hypotheses(h.flip(1), w.flip(1))
    torch.testing.assert_close(a, mean)
    torch.testing.assert_close(b, variance)


def test_initial_baseline_identity_and_no_logit_residual():
    torch.manual_seed(7)
    base = FixtureBaseline().eval()
    images = torch.rand(2, 3, 32, 32)
    expected = base(images).detach()
    model = RegionShapeSAM2(base, settings()).eval()
    actual = model.forward_details(images)
    torch.testing.assert_close(actual['logits'], expected)
    assert actual['occupancy'].shape == (2, 3, 8, 8)
    assert actual['weights'].shape == (2, 3)
    torch.testing.assert_close(actual['weights'].sum(1), torch.ones(2))
    # A forced decoder output must be the actual result, even if the initial
    # regional logits are huge. This detects z0 + delta implementations.
    base.override_initial = 100.
    base.sam.sam_mask_decoder.forced_output = 2.
    actual = model.forward_details(images)
    torch.testing.assert_close(actual['logits'], torch.full_like(actual['logits'], 2.))
    assert (actual['baseline_logits'] == 100).all()


def test_losses_reach_queries_compatibility_and_prompt_adapters():
    torch.manual_seed(19)
    model = RegionShapeSAM2(FixtureBaseline(), settings())
    result = model.forward_details(torch.rand(2, 3, 32, 32))
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:21, 10:25] = 1
    losses = region_shape_loss(result, target, torch.ones_like(target), model.config)
    assert all(torch.isfinite(v) for v in losses.values())
    losses['total'].backward()
    for param in (model.head.queries, model.head.compatibility[-1].weight,
                  model.dense_prompt[-1].weight, model.sparse_prompt.weight):
        assert param.grad is not None and param.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.baseline.parameters())
    assert not model.baseline.training


def test_auxiliary_labels_preserve_tiny_targets_and_ignore_invalid_pixels():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 1, 1] = 1
    coarse, _, sdf = coarse_targets(target, torch.ones_like(target), (8, 8), 8.)
    assert coarse.sum() == 1 and sdf.min() < 0
    for value in (0., 1.):
        _, _, field = coarse_targets(torch.full_like(target, value), torch.ones_like(target), (8, 8), 8.)
        assert torch.isfinite(field).all() and (field == (1 if value == 0 else -1)).all()
    model = RegionShapeSAM2(FixtureBaseline(), settings()).eval()
    result = model.forward_details(torch.rand(1, 3, 32, 32))
    valid = torch.ones_like(target)
    valid[:, :, :, 16:] = 0
    a = region_shape_loss(result, target, valid, model.config)
    changed = target.clone()
    changed[:, :, :, 16:] = 1
    b = region_shape_loss(result, changed, valid, model.config)
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_exchange_and_competition_ablations_have_distinct_behavior():
    torch.manual_seed(11)
    pyramid = [torch.randn(1, c, s, s) for c, s in [(4, 16), (8, 8), (16, 4)]]
    initial = torch.randn(1, 1, 32, 32)
    c = settings(mode='no_exchange')
    head = RegionShapeHead([4, 8, 16], c)
    no_exchange = head(pyramid, initial)
    expected = F.interpolate(initial, (8, 8), mode='bilinear', align_corners=False)
    torch.testing.assert_close(no_exchange['region'], expected)
    head.config = replace(c, mode='full')
    full = head(pyramid, initial)
    assert not torch.allclose(full['region'], no_exchange['region'])
    head.config = replace(c, mode='uniform_competition')
    uniform = head(pyramid, initial)
    torch.testing.assert_close(uniform['weights'], torch.full_like(uniform['weights'], 1/3))
    head.config = replace(c, mode='region_only')
    before = head(pyramid, initial)
    with torch.no_grad():
        head.queries.add_(10)
    after = head(pyramid, initial)
    torch.testing.assert_close(before['region'], after['region'])
    torch.testing.assert_close(before['mean'], after['mean'])


def test_mechanism_config_validation(tmp_path):
    path = tmp_path/'mechanism.yaml'
    path.write_text('unexpected: 1\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_mechanism_config(path)
    for config in [RegionShapeConfig(width=7), RegionShapeConfig(temperature=float('nan')),
                   RegionShapeConfig(rounds=0), RegionShapeConfig(mode='not-a-mode')]:
        with pytest.raises(ValueError):
            config.validate()


def test_training_roundtrip_bound_checkpoint_and_probe(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    test = make_manifest(tmp_path, 'test.csv', ['dd'])
    source = tmp_path/'pretrained.pt'
    source.write_bytes(b'workflow fixture only')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, batch_size=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path/'base')))
    monkeypatch.setattr(engine, 'build_model', lambda cfg, device: FixtureBaseline().to(device))
    baseline = engine.train(cfg)/'best.pt'
    original = baseline.read_bytes()
    out = workflow.train_region_shape(cfg, baseline, settings(), tmp_path/'mechanism', epochs=2)
    saved = workflow.read_artifact(out/'last_region_shape.pt')
    assert not any(name.startswith('baseline.') for name in saved['adapter'])
    assert baseline.read_bytes() == original
    with pytest.raises(ValueError, match='Not a baseline'):
        engine.load_adapter(FixtureBaseline(), out/'last_region_shape.pt', cfg)
    loaded, device, _ = workflow.build_system(cfg, baseline, checkpoint=out/'last_region_shape.pt')
    loaded.eval()
    assert torch.isfinite(loaded(torch.rand(1, 3, 32, 32))).all()
    scores = workflow.evaluate_region_shape(cfg, baseline, out/'last_region_shape.pt', test,
                                            tmp_path/'evaluation')
    assert scores['region_shape']['samples'] == 1
    inference = tmp_path/'inference.csv'
    with inference.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'image', 'mask', 'fov'])
        writer.writerow(['new-id', 'dd.png', 'nonexistent_mask.png', 'nonexistent_fov.png'])
    output = workflow.predict_region_shape(cfg, baseline, out/'last_region_shape.pt', inference,
                                            tmp_path/'prediction')
    assert (output/'000000.png').exists()
    modified = tmp_path/'changed.pt'
    modified.write_bytes(b'not the same baseline')
    with pytest.raises(ValueError, match='Different baseline'):
        workflow.build_system(cfg, modified, checkpoint=out/'last_region_shape.pt')
    wrong = replace(cfg, training=replace(cfg.training, threshold=.6))
    with pytest.raises(ValueError, match='settings differ'):
        workflow.build_system(wrong, baseline, checkpoint=out/'last_region_shape.pt')
    probe = workflow.probe_region_shape(cfg, baseline, settings(), tmp_path/'probe', steps=4)
    report = json.loads((probe/'probe.json').read_text())
    assert len(report['history']) == 4
    assert report['history'][0]['gradient_norms']['head'] > 0
    assert not list(probe.glob('*.pt'))
    assert baseline.read_bytes() == original
    stage_config = replace(settings(), multiscale_refinement=True,
                           refinement_only=True, refinement_width=4)
    with pytest.raises(ValueError, match='initialize-from'):
        workflow.train_region_shape(cfg, baseline, stage_config, tmp_path/'missing_init', epochs=1)
    stage = workflow.train_region_shape(cfg, baseline, stage_config, tmp_path/'stage', epochs=1,
                                       initialize_from=out/'last_region_shape.pt')
    stage_model, _, _ = workflow.build_system(cfg, baseline, checkpoint=stage/'last_region_shape.pt')
    assert all(p.requires_grad == name.startswith('refinement.')
               for name, p in stage_model.named_parameters())
    assert workflow.evaluate_region_shape(cfg, baseline, stage/'last_region_shape.pt', test,
        tmp_path/'stage_evaluation')['region_shape']['samples'] == 1
    record = workflow.read_manifest(train)[0]
    with workflow.Image.open(record.mask) as image:
        array = np.array(image, copy=True)
    array[0, 0] = 255-array[0, 0]
    workflow.Image.fromarray(array).save(record.mask)
    with pytest.raises(ValueError, match='data changed'):
        workflow.evaluate_region_shape(cfg, baseline, out/'last_region_shape.pt', test,
                                      tmp_path/'changed_data_evaluation')


@pytest.mark.integration
def test_real_sam2_joint_prompt_interface_identity_gradients_and_batching():
    pytest.importorskip('sam2')
    from sam2.build_sam import build_sam2

    from medsam_research.model import AutoSAM2

    torch.set_num_threads(2)
    torch.manual_seed(4)
    sam = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', ckpt_path=None, device='cpu',
                    apply_postprocessing=False, hydra_overrides_extra=['++model.image_size=64'])
    baseline = AutoSAM2(sam).eval()
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        reference = baseline(images)
    model = RegionShapeSAM2(baseline, settings()).eval()
    with torch.no_grad():
        result = model.forward_details(images)
        torch.testing.assert_close(result['logits'], reference, atol=1e-5, rtol=1e-5)
        singles = torch.cat([model(x[None]) for x in images])
        torch.testing.assert_close(result['logits'], singles, atol=1e-4, rtol=1e-4)
    model.train()
    result = model.forward_details(images)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 14:41, 18:47] = 1
    region_shape_loss(result, target, torch.ones_like(target), model.config)['total'].backward()
    assert model.sparse_prompt.weight.grad.abs().sum() > 0
    assert model.head.queries.grad.abs().sum() > 0
    assert all(p.grad is None for p in baseline.parameters())
    tuned = RegionShapeSAM2(baseline, replace(settings(), train_decoder=True))
    result = tuned.forward_details(images[:1])
    region_shape_loss(result, target[:1], torch.ones_like(target[:1]), tuned.config)['total'].backward()
    assert baseline.sam.sam_mask_decoder.conv_s0.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in baseline.sam.image_encoder.parameters())


@pytest.mark.parametrize('hypotheses', [1, 4])
def test_hypothesis_count_and_rounds_are_configurable(hypotheses):
    c = replace(settings(), hypotheses=hypotheses, rounds=1)
    model = RegionShapeSAM2(FixtureBaseline(), c)
    result = model.forward_details(torch.rand(1, 3, 32, 32))
    assert result['field'].shape == (1, hypotheses, 8, 8)
    assert len(result['history']) == 1
    target = torch.zeros(1, 1, 32, 32)
    loss = region_shape_loss(result, target, torch.ones_like(target), c)
    assert torch.isfinite(loss['total'])
    if hypotheses == 1:
        assert loss['diversity'] == 0
        torch.testing.assert_close(result['variance'], torch.zeros_like(result['variance']))
