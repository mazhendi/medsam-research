import csv
import json

import pytest
import torch
from test_data_metrics import make_manifest
from test_workflow import TinySegmenter

from medsam_research import engine
from medsam_research.config import Config, DataConfig, ModelConfig, TrainingConfig
from medsam_research.granularity import (
    FeatureAdapter,
    SupervisionController,
    evaluate_granularity,
    meta_objective,
    query_candidates,
    regional_losses,
    train_granularity,
)


def sample(seed):
    generator = torch.Generator().manual_seed(seed)
    return {'feature': torch.randn(1, 3, 4, 4, generator=generator),
            'baseline': torch.randn(1, 1, 8, 8, generator=generator),
            'target': torch.rand(1, 1, 8, 8, generator=generator).gt(.7).float(),
            'valid': torch.ones(1, 1, 8, 8)}


def test_occupancy_ignores_within_region_position_but_pixel_loss_does_not():
    y = torch.tensor([[[[1., 0.], [0., 0.]]]])
    logits = torch.tensor([[[[3., -3.], [-3., -3.]]]])
    p1, c1, _ = regional_losses(logits, y, torch.ones_like(y), 1)
    p2, c2, _ = regional_losses(logits.flip(-1), y, torch.ones_like(y), 1)
    torch.testing.assert_close(c1, c2)
    assert p2 > p1
    # Completely masked blocks do not contribute; changing their logits has no effect.
    valid = torch.tensor([[[[1., 0.], [0., 0.]]]])
    a, b, active = regional_losses(logits, y, valid, 2)
    assert active.sum() == 1
    assert torch.isfinite(a[active]).all() and torch.isfinite(b[active]).all()


@pytest.mark.parametrize('mode', ['meta', 'reweight'])
def test_controller_hypergradient_matches_finite_difference(mode):
    torch.manual_seed(17)
    head = FeatureAdapter(3, hidden=4)
    control = SupervisionController(3, grid=2)
    support, query = sample(1), sample(2)
    loss = meta_objective(head, control, support, query, 2, .5, mode)
    parameter = control.net[-1].bias if mode == 'meta' else control.net[-1].weight
    gradient = torch.autograd.grad(loss, parameter)[0]
    flat_index = gradient.abs().reshape(-1).argmax().item()
    analytic = gradient.reshape(-1)[flat_index].item()
    assert abs(analytic) > 1e-8
    epsilon = .02
    original = parameter.detach().clone()
    values = []
    for delta in (epsilon, -epsilon):
        with torch.no_grad():
            parameter.copy_(original)
            parameter.reshape(-1)[flat_index] += delta
        values.append(meta_objective(head, control, support, query, 2, .5, mode).item())
    with torch.no_grad():
        parameter.copy_(original)
    finite = (values[0]-values[1])/(2*epsilon)
    assert finite == pytest.approx(analytic, rel=.08, abs=3e-6)


def test_group_pairs_require_different_subject_same_dataset():
    values = [{'dataset': 'retina', 'group': 'a'}, {'dataset': 'retina', 'group': 'a'},
              {'dataset': 'retina', 'group': 'b'}]
    assert query_candidates(values) == {0: [2], 1: [2], 2: [0, 1]}
    with pytest.raises(ValueError, match='two different'):
        query_candidates(values[:2])
    with pytest.raises(ValueError, match='group'):
        query_candidates([{'dataset': 'retina', 'group': ''}])


@pytest.mark.parametrize('mode', ['pixel', 'fixed', 'curriculum', 'meta', 'reweight'])
def test_training_and_fixed_adapter_evaluation(tmp_path, monkeypatch, mode):
    class FeatureTiny(TinySegmenter):
        def forward_with_features(self, image):
            return self(image), image

    train = make_manifest(tmp_path, 'train.csv', ['aa', 'bb'])
    val = make_manifest(tmp_path, 'val.csv', ['cc'])
    # Fixture groups must represent distinct subjects for the meta pair protocol.
    with train.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for i, row in enumerate(rows):
        row['dataset'], row['group'] = 'fixture', f'person{i}'
    with train.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source = tmp_path/'source.pt'
    source.write_bytes(b'test source')
    cfg = Config(ModelConfig(checkpoint=str(source)), DataConfig(str(train), str(val)),
                 TrainingConfig(epochs=1, num_workers=0, device='cpu', amp=False,
                                output_dir=str(tmp_path/'baseline')))
    monkeypatch.setattr(engine, 'build_model', lambda cfg, device: FeatureTiny().to(device))
    baseline = engine.train(cfg)/'best.pt'
    original = baseline.read_bytes()
    out = tmp_path/mode
    train_granularity(cfg, baseline, out, mode=mode, epochs=2, grid=4)
    initial = json.loads((out/'initial_validation.json').read_text())
    assert initial['global'] == initial['adapted']
    assert baseline.read_bytes() == original
    saved = torch.load(out/'last_granularity.pt', weights_only=True)
    assert saved['adapter']['net.2.weight'].abs().sum() > 0
    history = [json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()]
    if mode in ('meta', 'reweight'):
        assert history[-1]['controller_gradient_norm'] > 0
        assert saved['controller']['net.2.weight'].abs().sum() > 0
    report = evaluate_granularity(cfg, baseline, out/'last_granularity.pt', val, tmp_path/'eval')
    assert 0 <= report['adapted']['dice'] <= 1
    assert len(list((tmp_path/'eval').glob('*_mask.png'))) == 2
    saved['provenance']['baseline_checkpoint_sha256'] = 'wrong'
    torch.save(saved, out/'wrong.pt')
    with pytest.raises(ValueError, match='different baseline'):
        evaluate_granularity(cfg, baseline, out/'wrong.pt', val, tmp_path/'bad')
