import torch
from torch import nn

from medsam_research.native_modules import RCSR, SBCA, SCDS, hierarchical_attention


class NativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.q_proj = nn.Linear(16, 8)
        self.k_proj = nn.Linear(16, 8)
        self.v_proj = nn.Linear(16, 8)
        self.out_proj = nn.Linear(8, 16)

    def _separate_heads(self, x, heads):
        return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)

    def forward(self, q, k, v):
        q, k, v = [self._separate_heads(layer(x), self.num_heads)
                   for layer, x in zip((self.q_proj, self.k_proj, self.v_proj), (q, k, v))]
        value = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out_proj(value.transpose(1, 2).flatten(2))


def test_hierarchy_handles_partial_windows_without_probability_leak():
    q, k, v = torch.randn(2, 2, 1, 4), torch.randn(2, 2, 49, 4), torch.randn(2, 2, 49, 4)
    result, mass = hierarchical_attention(q, k, v, 7, 7, 4)
    assert result.shape == (2, 2, 4)
    assert torch.isfinite(result).all()
    assert torch.allclose(mass.sum(-1), torch.ones(2, 2), atol=1e-6)
    assert (mass >= 0).all()


def test_sbca_preserves_other_tokens_and_supervises_background():
    module = SBCA(NativeAttention(), mask_query_index=2, windows=(2, 4))
    q, k, v = torch.randn(2, 6, 16), torch.randn(2, 49, 16), torch.randn(2, 49, 16)
    native = module.base(q, k, v)
    assert torch.equal(module(q, k, v), native)
    module.gain.data.fill_(.5)
    result = module(q, k, v)
    keep = [0, 1, 3, 4, 5]
    assert torch.equal(result[:, keep], native[:, keep])
    assert not torch.allclose(result[:, 2], native[:, 2])
    evidence = module.last_evidence
    loss = result.square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(
        evidence, torch.rand_like(evidence))
    loss.backward()
    assert module.background_query.grad.abs().sum() > 0
    assert module.scale_gate.weight.grad.abs().sum() > 0


def test_scds_identity_native_channels_and_deep_gradients():
    torch.set_num_threads(2)
    module = SCDS(12, (16, 24), width=8, pool_size=4)
    shallow = torch.randn(2, 12, 13, 13)
    deep = [torch.randn(2, 16, 7, 7, requires_grad=True),
            torch.randn(2, 24, 3, 3, requires_grad=True)]
    assert torch.equal(module(shallow, deep), shallow)
    module.gain.data.fill_(.2)
    module(shallow, deep).square().mean().backward()
    assert all(feature.grad.abs().sum() > 0 for feature in deep)
    assert module.attention.in_proj_weight.grad.abs().sum() > 0


def test_rcsr_identity_and_region_conditioned_sampling():
    module = RCSR(channels=8, token_dim=16)
    x, high, token = torch.randn(2, 8, 11, 13), torch.randn(2, 8, 11, 13), torch.randn(2, 16)
    assert torch.equal(module(x, high, token), x)
    module.gain.data.fill_(.5)
    nn.init.normal_(module.predict[-1].weight, std=.01)
    result = module(x, high, token)
    assert result.shape == x.shape
    assert not torch.allclose(result, module(x, high, token + 2))
    result.square().mean().backward()
    assert module.condition[0].weight.grad.abs().sum() > 0
    assert module.predict[-1].weight.grad.abs().sum() > 0
