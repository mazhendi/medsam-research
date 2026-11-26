"""Candidate B: local-content step sizes, multiscale-conditioned SSM read/write."""
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .selective_mamba import SelectiveMamba1, _cuda_scan, selective_scan_torch


class ReadWriteConditionedMamba(SelectiveMamba1):
    """Mamba-1 recurrence with B/C from a separate same-grid condition stream.

    Content u, output gate z and delta retain the local sequence path. A and D
    retain the standard parameterization. This is a modification, not an
    official Mamba implementation or an additional attention block.
    """
    def __init__(self, width, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__(width, d_state, d_conv, expand, backend)
        # Original joint dt/B/C projection must not remain as unused parameters.
        del self.x_proj
        self.content_dt = nn.Linear(self.inner, self.rank, bias=False)
        self.condition_bc = nn.Linear(width, 2 * d_state, bias=False)

    def scan_inputs(self, tokens, condition):
        if tokens.shape != condition.shape:
            raise ValueError("Content and condition must have identical B,L,C shapes")
        x, z = self.in_proj(tokens).chunk(2, -1)
        length = tokens.shape[1]
        u = F.silu(self.conv(x.transpose(1, 2))[..., :length])
        dt = self.content_dt(u.transpose(1, 2))
        delta = F.linear(dt, self.dt_proj.weight).transpose(1, 2).contiguous()
        b, c = self.condition_bc(condition).chunk(2, dim=-1)
        return u, z, delta, b.transpose(1, 2).contiguous(), c.transpose(1, 2).contiguous()

    def forward(self, tokens, condition):
        u, z, delta, b, c = self.scan_inputs(tokens, condition)
        a = -self.A_log.exp()
        if self.backend == "fused" and not tokens.is_cuda:
            raise RuntimeError("fused Mamba scan requires CUDA; use auto or torch on CPU")
        scan = _cuda_scan() if tokens.is_cuda and self.backend != "torch" else None
        if scan is not None:
            y = scan(u.contiguous(), delta, a, b, c, self.D, z=None,
                     delta_bias=self.dt_proj.bias, delta_softplus=True)
            self.last_backend = "fused"
        else:
            if self.backend == "fused":
                raise RuntimeError("mamba-ssm CUDA scan unavailable; install a compatible build")
            if tokens.is_cuda and self.backend == "auto":
                warnings.warn("Using portable read/write-conditioned selective scan; "
                              "check memory/time with probe-abc-pvt before training.", stacklevel=2)
            y = selective_scan_torch(u, delta, a, b, c, self.D, self.dt_proj.bias)
            self.last_backend = "torch"
        return self.out_proj(y.transpose(1, 2) * F.silu(z))


class MultiscaleReadWriteB(nn.Module):
    """Full-width DW1/3/5/7 condition B/C, one raster SSM, identity residual."""
    def __init__(self, width=32, d_state=16, d_conv=4, expand=2, backend="auto", residual=True):
        super().__init__()
        self.residual = residual
        self.project = nn.Conv2d(width, width, 1)
        self.scales = nn.ModuleList(nn.Conv2d(width, width, k, padding=k // 2, groups=width)
                                    for k in (1, 3, 5, 7))
        self.condition_mix = nn.Conv2d(4 * width, width, 1)
        self.condition_norm = nn.LayerNorm(width)
        self.mamba = ReadWriteConditionedMamba(width, d_state, d_conv, expand, backend)
        self.output = nn.Conv2d(width, width, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        p = self.project(x)
        condition = self.condition_mix(torch.cat([layer(p) for layer in self.scales], dim=1))
        condition = self.condition_norm(condition.flatten(2).transpose(1, 2))
        tokens = p.flatten(2).transpose(1, 2)
        mixed = self.mamba(tokens, condition).transpose(1, 2).reshape(batch, channels, height, width)
        output = self.output(mixed)
        return x + output if self.residual else output
