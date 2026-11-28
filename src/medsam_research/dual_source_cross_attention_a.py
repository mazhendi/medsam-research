"""Dual-source values with one shared attention map and an aligned deep skip."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .multiscale_cross_attention_a import MultiscaleCrossAttentionA


class DualSourceCrossAttentionA(MultiscaleCrossAttentionA):
    """Y = X + resize(skip(G)) + output(cat(W Vx, W Vg)).

    Q/K preparation is identical to MultiscaleCrossAttentionA. W is computed
    once per head and shared by both value streams. No learned residual scales.
    """
    def __init__(self, channels, context_channels, width=32, heads=4):
        super().__init__(channels, context_channels, width, heads)
        self.shallow_value = nn.Conv2d(channels, width, 1)
        self.context_skip = nn.Conv2d(context_channels, channels, 1)
        self.output = nn.Conv2d(2*width, channels, 1)

    def dual_attention(self, q, k, vx, vg):
        b,c,h,w=q.shape
        d=c//self.heads
        query=q.reshape(b,self.heads,d,h*w)
        keys=F.unfold(k,5,padding=2).reshape(b,self.heads,d,25,h*w)
        scores=(query.unsqueeze(3)*keys).sum(2)*d**-0.5
        valid=F.unfold(q.new_ones(1,1,h,w),5,padding=2).bool()
        if scores.dtype in (torch.float16,torch.bfloat16):
            scores=scores.float()
        weights=scores.masked_fill(~valid.unsqueeze(1),float('-inf')).softmax(2)
        # Both sources use exactly the same weights, without a second softmax.
        results=[]
        for value in (vx,vg):
            neighbors=F.unfold(value,5,padding=2).reshape(b,self.heads,d,25,h*w)
            results.append((weights.to(value.dtype).unsqueeze(2)*neighbors).sum(3).reshape(b,c,h,w))
        return torch.cat(results,dim=1)

    def forward(self, x, context):
        normalized=self.norm_x(x)
        q=self.query(normalized)
        vx=self.shallow_value(normalized)
        p=self.context_project(context)
        h=self.context_fuse(torch.cat([branch(p) for branch in self.scales],dim=1))
        h=self.norm_h(F.interpolate(h,size=x.shape[-2:],mode='bilinear',align_corners=False))
        k,vg=self.key(h),self.value(h)
        if self.training and torch.is_grad_enabled():
            result=checkpoint(self.dual_attention,q,k,vx,vg,use_reentrant=False)
        else:
            result=self.dual_attention(q,k,vx,vg)
        skip=F.interpolate(self.context_skip(context),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return x + skip + self.output(result)
