import torch

from medsam_research.adapter_probe import fit_probe


def test_probe_demonstrates_fitting_and_zero_initialized_layer_gradient_delay():
    # Learnable feature/label relationship: fitting needs feature-dependent corrections.
    torch.manual_seed(2)
    features = torch.randn(2, 3, 8, 8)
    sample = {'feature': features, 'baseline': torch.zeros(2, 1, 8, 8),
              'target': (features[:, :1] > 0).float(), 'valid': torch.ones(2, 1, 8, 8)}
    history, layers, error = fit_probe(sample, 3, 60, 'adam', .03, 42, .5)
    assert error is None
    assert history[-1]['loss'] < history[0]['loss'] * .7
    assert history[-1]['dice_model_grid'] > history[0]['dice_model_grid']
    assert history[-1]['parameter_drift_norm'] > 0
    assert history[-1]['residual_abs_mean'] > .1
    first = {r['parameter']: r for r in layers if r['step'] == 1}
    assert first['net.0.weight']['gradient_norm'] == 0  # Expected at the first update only.
    assert first['net.2.weight']['gradient_norm'] > 0
    final = {r['parameter']: r for r in layers if r['step'] == 60}
    assert final['net.0.weight']['gradient_norm'] > 0
    repeated, _, _ = fit_probe(sample, 3, 60, 'adam', .03, 42, .5)
    assert repeated == history


def test_probe_reports_divergence_without_nonfinite_json():
    sample = {'feature': torch.ones(1, 3, 4, 4)*1e20,
              'baseline': torch.zeros(1, 1, 4, 4),
              'target': torch.ones(1, 1, 4, 4), 'valid': torch.ones(1, 1, 4, 4)}
    history, _, error = fit_probe(sample, 3, 5, 'sgd', 1e30, 42, .5)
    assert error is not None
    assert history[0]['step'] == 0
