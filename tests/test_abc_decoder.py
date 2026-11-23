from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from medsam_research.abc_decoder import (
    ABCPVT,
    ChannelSpatialC,
    LocalMambaB,
    SpatialContextA,
    abc_loss,
)
from medsam_research.polyp_pvt_workflow import load_pvt_config
from medsam_research.selective_mamba import SelectiveMamba1, selective_scan_torch


class FakePVT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(3, c, 1) for c in (64, 128, 320, 512))

    def forward(self, x):
        return [layer(F.avg_pool2d(x, stride)) for layer, stride in
                zip(self.layers, (4, 8, 16, 32))]


def sequential_scan(u, delta, a, b, c, d, bias):
    h = torch.zeros(u.shape[0], u.shape[1], a.shape[1], dtype=u.dtype)
    output = []
    for index in range(u.shape[-1]):
        dt = F.softplus(delta[..., index]+bias)
        h = (torch.exp(dt[..., None]*a)*h
             + dt[..., None]*b[:, None, :, index]*u[..., index, None])
        output.append((h*c[:, None, :, index]).sum(-1)+d*u[..., index])
    return torch.stack(output, -1)


@pytest.mark.parametrize("length", [1, 5, 8, 13])
def test_parallel_scan_matches_recurrence_and_gradients(length):
    torch.manual_seed(10)
    shape = (2, 3, length)
    values = [torch.randn(shape, dtype=torch.float64), torch.randn(shape, dtype=torch.float64),
              -torch.rand(3, 2, dtype=torch.float64),
              torch.randn(2, 2, length, dtype=torch.float64),
              torch.randn(2, 2, length, dtype=torch.float64),
              torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64)]
    first = [v.clone().requires_grad_() for v in values]
    second = [v.clone().requires_grad_() for v in values]
    parallel, serial = selective_scan_torch(*first), sequential_scan(*second)
    torch.testing.assert_close(parallel, serial, rtol=1e-10, atol=1e-10)
    grads1 = torch.autograd.grad(parallel.square().sum(), first, allow_unused=True)
    grads2 = torch.autograd.grad(serial.square().sum(), second)
    for g1, g2, value in zip(grads1, grads2, first):
        # At L=1 and zero initial state, A has mathematically zero influence.
        g1 = torch.zeros_like(value) if g1 is None else g1
        torch.testing.assert_close(g1, g2, rtol=1e-9, atol=1e-9)


def test_mamba_is_causal_and_all_parameters_receive_gradients():
    module = SelectiveMamba1(8, d_state=4, backend="torch").eval()
    tokens = torch.randn(2, 11, 8)
    modified = tokens.clone()
    modified[:, 6:] += 10
    torch.testing.assert_close(module(tokens)[:, :6], module(modified)[:, :6])
    module(tokens).square().mean().backward()
    assert module.last_backend == "torch"
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert module.A_log.grad.norm() > 0
    with pytest.raises(RuntimeError, match="CUDA"):
        SelectiveMamba1(8, backend="fused")(tokens)


def test_a_keeps_shape_and_trains_all_spatial_context_paths():
    module = SpatialContextA(16)
    image = torch.randn(2, 16, 12, 16)
    output = module(image)
    assert output.shape == image.shape
    output.square().mean().backward()
    assert all(branch.weight.grad.norm() > 0 for branch in module.context)
    assert module.spatial[0].weight.grad.norm() > 0


def test_b_progressive_links_use_completed_previous_group():
    module = LocalMambaB(16, d_state=2, expand=1, backend="torch")
    module.project = nn.Identity()
    module.fuse = nn.Identity()
    for collection in (module.local, module.norm, module.mamba, module.mix):
        for index in range(4):
            collection[index] = nn.Identity()
    x = torch.cat([torch.full((1, 4, 2, 2), float(i)) for i in (1, 2, 3, 4)], 1)
    expected = torch.cat([torch.full((1, 4, 2, 2), float(i)) for i in (2, 5, 9, 14)], 1)
    torch.testing.assert_close(module(x), expected)


def test_c_weight_shape_sum_and_two_selection_branches():
    module = ChannelSpatialC(16)
    x, y = torch.randn(2, 16, 12, 16), torch.randn(2, 16, 3, 4)
    px, py, weights = module.prepare(x, y)
    assert px.shape == py.shape == x.shape
    assert weights.shape == (2, 2, 16, 12, 16)
    torch.testing.assert_close(weights.sum(1), torch.ones_like(x))
    module(x, y).square().mean().backward()
    assert module.global_select[-1].weight.grad.norm() > 0
    assert module.local_select[-1].weight.grad.norm() > 0


def test_c_selects_original_projected_stream_not_joint_descriptor():
    module = ChannelSpatialC(8).eval()
    with torch.no_grad():
        for layer in (module.fine, module.deep, module.output):
            layer.weight.zero_()
            layer.weight[:, :, 0, 0].copy_(torch.eye(8))
            layer.bias.zero_()
        module.global_select[-1].weight.zero_()
        module.global_select[-1].bias.copy_(torch.tensor([20.]*8+[-20.]*8))
        module.local_select[-1].weight.zero_()
        module.local_select[-1].bias.zero_()
    fine, deep = torch.randn(2, 8, 8, 10), torch.randn(2, 8, 4, 5)
    torch.testing.assert_close(module(fine, deep), fine)


@pytest.mark.parametrize("a,b,c", [(False,False,False),(True,False,False),
    (False,True,False),(False,False,True),(True,True,False),(True,False,True),
    (False,True,True),(True,True,True)])
def test_abc_all_combinations_forward_backward_and_decoder_order(a,b,c):
    model = ABCPVT(FakePVT(), width=16, a=a, b=b, c=c, d_state=2, expand=1, backend="torch")
    seen = []
    for index, decoder in enumerate(model.decoder):
        decoder.register_forward_pre_hook(lambda _, args, i=index:
                                           seen.append((i, args[0].shape[-1], args[1].shape[-1])))
    outputs = model(torch.randn(2, 3, 64, 64))
    assert seen == [(2, 4, 2), (1, 8, 4), (0, 16, 8)]
    assert set(outputs) == {"final", "aux2", "aux3"}
    assert all(v.shape == (2, 1, 64, 64) for v in outputs.values())
    loss = abc_loss(outputs, (torch.rand(2, 1, 64, 64)>.7).float())
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_abc_configs_use_fixed_protocol_with_unique_outputs():
    root = Path(__file__).resolve().parents[1]
    configs = [load_pvt_config(root / f"configs/abc_pvt_{name}.yaml")
               for name in ("full", "control", "a_only", "b_only", "c_only", "ab", "ac", "bc")]
    assert len({cfg["output"] for cfg in configs}) == 8
    for cfg in configs:
        assert cfg["architecture"] == "abc_pvt_b2" and cfg["image_size"] == 352
        assert cfg["batch_size"] == 16 and cfg["epochs"] == 99
        assert cfg["decoder"]["width"] == 32 and cfg["decoder"]["backend"] == "auto"
