from pathlib import Path

import pytest
import torch


@pytest.mark.integration
def test_official_biomedclip_patch_text_backend():
    """Optional real weights test. Never downloads weights during pytest/CI."""
    root = Path(__file__).resolve().parents[1]/'checkpoints'/'biomedclip'
    if not (root/'bundle.json').exists():
        pytest.skip('Prepare the pinned BiomedCLIP bundle for this optional integration test')
    pytest.importorskip('open_clip')
    from medsam_research.text_evidence import FrozenBiomedCLIP

    torch.set_num_threads(2)
    model = FrozenBiomedCLIP(root, torch.device('cpu'))
    model.train()  # Frozen backend must remain deterministic and in eval mode.
    image = torch.rand(1, 3, 64, 64).repeat(2, 1, 1, 1)
    response, text = model(image, ['retinal blood vessels', 'colorectal polyp'])
    assert response.shape == (2, 1, 14, 14) and text.shape == (2, 512)
    assert torch.isfinite(response).all() and torch.isfinite(text).all()
    torch.testing.assert_close(text.norm(dim=-1), torch.ones(2))
    assert not torch.allclose(response[0], response[1])
    assert not model.training and all(not p.requires_grad for p in model.parameters())

    model.sampling_mode, model.tile_grid, model.tile_batch_size = 'tiled', 2, 3
    tiled, same_text = model(image[:1], ['retinal blood vessels'])
    assert tiled.shape == (1, 1, 28, 28) and torch.isfinite(tiled).all()
    torch.testing.assert_close(same_text, text[:1])
