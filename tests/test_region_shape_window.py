"""Window-routing algebra, old checkpoint compatibility and diagnostic isolation."""
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_data_metrics import make_manifest
from test_region_shape import FixtureBaseline, settings

from medsam_research import engine
from medsam_research import region_shape_training as workflow
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig
from medsam_research.region_shape import RegionShapeHead, RegionShapeSAM2, region_shape_loss
from medsam_research.region_shape_diagnostics import (
    diagnose_region_shape,
    dice,
    error_auc,
    hypothesis_report,
    window_oracle,
)


def test_window_selection_uses_spatial_evidence_and_preserves_simplex():
    torch.set_num_threads(2)
    c = replace(settings(), rounds=1, routing='window', window_size=4, local_strength=1.)
    head = RegionShapeHead([4, 8, 16], c)
    # Remove the learned term to isolate the predicted consistency signal.
    with torch.no_grad():
        for p in head.compatibility.parameters():
            p.zero_()
    occupancy = torch.zeros(1, 2, 8, 8)
    occupancy[:, 0] = 1
    probability = torch.full((1, 1, 8, 8), .1)
    probability[:, :, :, :4] = .9
    feature, queries = torch.randn(1, 8, 8, 8), torch.randn(1, 2, 8)
    global_w, mean, variance, _, spatial = head.competition(feature, probability, queries,
                                                          occupancy, return_spatial=True)
    torch.testing.assert_close(global_w, torch.full_like(global_w, .5))
    assert mean[0, 0, 2, 1] > .85 and mean[0, 0, 2, 6] < .15
    torch.testing.assert_close(spatial.sum(1), torch.ones(1, 8, 8))
    assert spatial.min() >= 0 and torch.all((variance >= 0) & (variance <= .25+1e-6))
    # Positive anchor strength moves window decisions toward the global mixture.
    head.config = replace(c, local_strength=.5)
    anchored = head.competition(feature, probability, queries, occupancy)[1]
    assert mean[0, 0, 2, 1] > anchored[0, 0, 2, 1] > .5


def test_zero_strength_global_equivalence_single_window_and_parameter_parity():
    torch.manual_seed(52)
    c = settings()
    global_head = RegionShapeHead([4, 8, 16], c)
    window_head = RegionShapeHead([4, 8, 16], replace(c, routing='window', window_size=4,
                                                     local_strength=0.))
    window_head.load_state_dict(global_head.state_dict(), strict=True)
    assert sum(p.numel() for p in global_head.parameters()) == sum(p.numel() for p in window_head.parameters())
    pyramid = [torch.randn(1, n, z, z) for n, z in [(4, 16), (8, 8), (16, 4)]]
    logits = torch.randn(1, 1, 32, 32)
    a, b = global_head(pyramid, logits), window_head(pyramid, logits)
    for key in ('field', 'region', 'weights', 'mean', 'variance'):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    window_head.config = replace(c, routing='window', window_size=8, local_strength=.5)
    one = window_head(pyramid, logits)
    for key in ('field', 'region', 'mean', 'variance'):
        torch.testing.assert_close(a[key], one[key], atol=2e-6, rtol=2e-6)


def test_window_permutation_equivariance_and_uniform_control():
    torch.manual_seed(3)
    c = replace(settings(), routing='window', window_size=4)
    head = RegionShapeHead([4, 8, 16], c)
    feature = torch.randn(1, 8, 8, 8)
    probability = torch.rand(1, 1, 8, 8)
    queries = torch.randn(1, 3, 8)
    occupancy = torch.rand(1, 3, 8, 8)
    a = head.competition(feature, probability, queries, occupancy, return_spatial=True)
    perm = torch.tensor([2, 0, 1])
    b = head.competition(feature, probability, queries[:, perm], occupancy[:, perm], return_spatial=True)
    torch.testing.assert_close(a[1], b[1])
    torch.testing.assert_close(a[2], b[2])
    torch.testing.assert_close(a[-1][:, perm], b[-1])
    head.config = replace(c, mode='uniform_competition')
    uniform = head.competition(feature, probability, queries, occupancy, return_spatial=True)
    torch.testing.assert_close(uniform[-1], torch.full_like(occupancy, 1/3))


def test_window_model_preserves_baseline_identity_and_auxiliary_gradient_paths():
    torch.set_num_threads(2)
    torch.manual_seed(14)
    c = replace(settings(), routing='window', window_size=4)
    base = FixtureBaseline().eval()
    images = torch.rand(2, 3, 32, 32)
    reference = base(images).detach()
    model = RegionShapeSAM2(base, c)
    out = model.forward_details(images)
    torch.testing.assert_close(out['logits'], reference)
    assert out['spatial_weights'].shape == (2, 3, 8, 8)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 9:22, 12:24] = 1
    loss = region_shape_loss(out, target, torch.ones_like(target), c)['total']
    loss.backward()
    assert model.head.queries.grad.abs().sum() > 0
    assert model.head.compatibility[-1].weight.grad.abs().sum() > 0
    assert model.dense_prompt[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize('seed', [14, 3, 24, 8, 71])
def test_oracle_matches_exhaustive_hard_window_selection_with_invalid_pixels(seed):
    rng = np.random.default_rng(seed)
    candidates = rng.random((2, 4, 4)) > .5
    target = rng.random((4, 4)) > .5
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    score, mask, _ = window_oracle(candidates, target, valid, (2, 2))
    scores = []
    for choices in itertools.product(range(2), repeat=4):
        selected = np.zeros_like(target)
        for i, k in enumerate(choices):
            y, x = (i//2)*2, (i % 2)*2
            selected[y:y+2, x:x+2] = candidates[k, y:y+2, x:x+2]
        scores.append(dice(selected, target, valid))
    assert score == max(scores) == dice(mask, target, valid)
    assert score >= max(dice(p, target, valid) for p in candidates)
    empty = np.zeros((4, 4), bool)
    assert window_oracle(np.stack((empty, ~empty)), empty, valid, (2, 2))[0] == 1


def test_diagnostic_auc_ties_and_candidate_collapse():
    errors = np.array([[0, 0], [1, 1]], bool)
    valid = np.ones_like(errors)
    assert error_auc(errors.astype(float), errors, valid) == 1
    assert error_auc(np.ones((2, 2)), errors, valid) == .5
    assert error_auc(np.ones((2, 2)), np.zeros_like(errors), valid) is None
    occupancy = torch.full((3, 4, 4), .8)
    spatial = torch.full_like(occupancy, 1/3)
    target = np.ones((8, 8), bool)
    report = hypothesis_report(occupancy, torch.zeros(1, 4, 4), spatial,
                                torch.full((3,), 1/3), target, target,
                                np.ones_like(target), .5, 2)
    assert report['mean_pairwise_iou'] == 1
    assert report['mean_pairwise_probability_l1'] == 0
    assert report['window_oracle_gain_over_best_candidate'] == 0
    assert report['disagreement_error_auroc'] is None


def test_old_checkpoint_normalization_new_training_roundtrip_and_diagnostic(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    source = tmp_path/'pretrained.pt'
    source.write_bytes(b'fixture only')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, batch_size=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path/'base')))
    def fixture_model(cfg, device):
        # Frozen fixture parameters must also be identical at reconstruction,
        # analogous to loading official pretrained SAM weights in real runs.
        with torch.random.fork_rng():
            torch.manual_seed(995)
            return FixtureBaseline().to(device)
    monkeypatch.setattr(engine, 'build_model', fixture_model)
    baseline = engine.train(cfg)/'best.pt'
    base, _ = engine._restore(cfg, baseline)
    original = RegionShapeSAM2(base, settings())
    old = workflow.artifact(original, cfg, baseline, 0, {}, [])
    for key in ('routing', 'window_size', 'local_strength'):
        del old['mechanism'][key]
    old_path = tmp_path/'old.pt'
    torch.save(old, old_path)
    restored, _, _ = workflow.build_system(cfg, baseline, settings(), checkpoint=old_path)
    assert restored.config.routing == 'global'
    restored.eval()
    images = torch.rand(1, 3, 32, 32)
    original.eval()
    torch.testing.assert_close(restored(images), original(images))
    c = replace(settings(), routing='window', window_size=4)
    output = workflow.train_region_shape(cfg, baseline, c, tmp_path/'window', epochs=1)
    checkpoint = output/'last_region_shape.pt'
    loaded, _, _ = workflow.build_system(cfg, baseline, checkpoint=checkpoint)
    assert loaded.config.routing == 'window'
    summary = diagnose_region_shape(cfg, baseline, checkpoint, val, tmp_path/'diagnosis', window_size=4)
    assert summary['samples'] == 1
    result = json.loads((tmp_path/'diagnosis/summary.json').read_text())
    assert 'never enter inference' in result['scope']
    assert (tmp_path/'diagnosis/per_image.json').exists()
    with pytest.raises(ValueError, match='divide'):
        diagnose_region_shape(cfg, baseline, checkpoint, val, tmp_path/'invalid', window_size=3)


def test_window_configuration_and_bf16_loss():
    torch.set_num_threads(2)
    for c in [replace(settings(), routing='invalid'),
              replace(settings(), routing='window', window_size=3),
              replace(settings(), local_strength=float('nan')),
              replace(settings(), local_strength=1.1)]:
        with pytest.raises(ValueError):
            c.validate()
    c = replace(settings(), routing='window', window_size=4)
    head = RegionShapeHead([4, 8, 16], c)
    pyramid = [torch.randn(1, n, z, z) for n, z in [(4, 16), (8, 8), (16, 4)]]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = head(pyramid, torch.randn(1, 1, 32, 32))
        loss = result['mean'].mean()+result['variance'].mean()+result['region'].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
