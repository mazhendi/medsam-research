import pytest
import torch
from torch.nn import functional as F

from medsam_research.text_evidence import tiled_response


def test_tiled_alignment_and_batch_independence():
    # Linear pixel-center coordinates must survive overlap blending without
    # seams, translations or mixing samples/texts between tile batches.
    axis = (torch.arange(112)+.5)/112
    plane = axis[:, None]+axis[None, :]
    images = plane[None, None].repeat(2, 3, 1, 1)
    text = torch.tensor([[0.], [3.]])

    def encode(crops, embeddings):
        return F.interpolate(crops[:, :1], (14, 14), mode='bilinear',
                             align_corners=False)+embeddings[:, :, None, None]

    a = tiled_response(images, text, encode, grid=2, batch_size=1)
    b = tiled_response(images, text, encode, grid=2, batch_size=5)
    expected = F.interpolate(images[:, :1], (28, 28), mode='bilinear',
                             align_corners=False)+text[:, :, None, None]
    assert a.shape == (2, 1, 28, 28)
    torch.testing.assert_close(a, expected)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize('grid', [2, 4])
def test_tiled_constant_response_covers_edges(grid):
    result = tiled_response(torch.ones(1, 3, 64, 64), torch.ones(1, 2),
                            lambda x, t: x.new_full((len(x), 1, 14, 14), .7), grid, 3)
    assert result.shape == (1, 1, 14*grid, 14*grid)
    torch.testing.assert_close(result, torch.full_like(result, .7))


def test_tiled_invalid_geometry():
    with pytest.raises(ValueError, match='divisible'):
        tiled_response(torch.ones(1, 3, 65, 64), torch.ones(1, 2), None)
