"""Small Mamba-1 mixer with an explicit selective SSM and optional CUDA scan.

Equations follow Mamba-1 (Gu and Dao, arXiv:2312.00752). This is a project
implementation, not a vendored or reproduced VMMCA-Net block. The portable scan
uses associative prefix composition; it is not replaced by convolution/attention.
"""
import math
import warnings
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def _cuda_scan():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except (ImportError, OSError):
        return None
    return selective_scan_fn


def selective_scan_torch(u, delta, a, b, c, d, delta_bias):
    """u,delta:B,D,L; a:D,N; b,c:B,N,L; d,bias:D. Zero initial state.

    h_t=exp(softplus(dt_t)*A)*h_(t-1)+softplus(dt_t)*B_t*u_t;
    y_t=C_t*h_t+D*u_t. Parallel prefix evaluation has O(log L) stages.
    """
    step = F.softplus(delta+delta_bias[None, :, None])
    decay = torch.exp(step[:, :, None, :]*a[None, :, :, None])
    state = step[:, :, None, :]*b[:, None, :, :]*u[:, :, None, :]
    stride = 1
    while stride < u.shape[-1]:
        previous_decay, previous_state = decay, state
        state = torch.cat((previous_state[..., :stride],
                           previous_state[..., stride:]
                           + previous_decay[..., stride:]*previous_state[..., :-stride]), -1)
        decay = torch.cat((previous_decay[..., :stride],
                           previous_decay[..., stride:]*previous_decay[..., :-stride]), -1)
        stride *= 2
    return (state*c[:, None, :, :]).sum(2)+d[None, :, None]*u


class SelectiveMamba1(nn.Module):
    def __init__(self, width, d_state=16, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        if backend not in {"auto", "torch", "fused"}:
            raise ValueError("Mamba backend must be auto, torch or fused")
        if min(width, d_state, d_conv, expand) < 1:
            raise ValueError("Mamba dimensions must be positive")
        self.backend, self.last_backend = backend, None
        self.inner, self.state_size = width*expand, d_state
        self.rank = max(1, math.ceil(width/16))
        self.in_proj = nn.Linear(width, self.inner*2, bias=False)
        self.conv = nn.Conv1d(self.inner, self.inner, d_conv, padding=d_conv-1,
                              groups=self.inner)
        self.x_proj = nn.Linear(self.inner, self.rank+2*d_state, bias=False)
        self.dt_proj = nn.Linear(self.rank, self.inner)
        nn.init.uniform_(self.dt_proj.weight, -self.rank**-.5, self.rank**-.5)
        dt = torch.exp(torch.empty(self.inner).uniform_(math.log(.001), math.log(.1)))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, d_state+1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def forward(self, tokens):
        x, z = self.in_proj(tokens).chunk(2, -1)
        length = tokens.shape[1]
        u = F.silu(self.conv(x.transpose(1, 2))[..., :length])
        coefficients = self.x_proj(u.transpose(1, 2))
        dt, b, c = torch.split(coefficients, [self.rank, self.state_size, self.state_size], -1)
        delta = F.linear(dt, self.dt_proj.weight).transpose(1, 2).contiguous()
        a = -self.A_log.exp()
        b, c = b.transpose(1, 2).contiguous(), c.transpose(1, 2).contiguous()
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
                warnings.warn("Using portable selective scan: mamba-ssm CUDA extension unavailable; "
                              "check memory/time with probe-abc-pvt before full training.",
                              stacklevel=2)
            y = selective_scan_torch(u, delta, a, b, c, self.D, self.dt_proj.bias)
            self.last_backend = "torch"
        return self.out_proj(y.transpose(1, 2)*F.silu(z))
