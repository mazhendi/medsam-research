"""Static-image, prompt-free SAM 2.1 baseline using learned task tokens.

Uses the official decoder directly: no predictor inference decorators, video memory,
ground-truth prompts, object-presence gating, or multimask oracle selection.
"""

import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import SAM2_COMMIT


def upstream_commit():
    import sam2

    directory = Path(sam2.__file__).resolve().parent
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class AutoSAM2(nn.Module):
    def __init__(self, sam, prompt_tokens=4):
        super().__init__()
        self.sam = sam
        self.sam.requires_grad_(False)
        self.sam.sam_mask_decoder.requires_grad_(True)
        dim = sam.sam_prompt_encoder.embed_dim
        self.task_tokens = nn.Parameter(torch.randn(1, prompt_tokens, dim) * 0.02)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.image_size = sam.image_size
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen modules deterministic; only the decoder and task tokens learn.
        self.sam.eval()
        self.sam.sam_mask_decoder.train(mode)
        return self

    def forward(self, images):
        return self.forward_with_features(images)[0]

    def forward_with_features(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected Bx3x{self.image_size}x{self.image_size}")
        x = (images - self.mean) / self.std
        # Do not wrap forward_image in no_grad: it includes trainable conv_s0/conv_s1.
        features = self.sam.forward_image(x)["backbone_fpn"][-self.sam.num_feature_levels :]
        embed = features[-1]
        if self.sam.directly_add_no_mem_embed:
            embed = embed + self.sam.no_mem_embed.reshape(1, -1, 1, 1)
        _, dense = self.sam.sam_prompt_encoder(points=None, boxes=None, masks=None)
        batch = images.shape[0]
        masks, _, _, _ = self.sam.sam_mask_decoder(
            image_embeddings=embed,
            image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.task_tokens.expand(batch, -1, -1),
            dense_prompt_embeddings=dense.expand(batch, -1, -1, -1),
            multimask_output=False,
            repeat_image=False,
            high_res_features=features[:-1] if len(features) > 1 else None,
        )
        logits = F.interpolate(
            masks.float(), size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits, embed


def build_model(cfg, device):
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError("Install the pinned official SAM 2 checkout; see README.md") from exc
    revision = upstream_commit()
    if revision != SAM2_COMMIT:
        raise RuntimeError(f"Expected SAM 2 commit {SAM2_COMMIT}; found {revision}")
    if not Path(cfg.checkpoint).is_file():
        raise FileNotFoundError(f"Missing official pretrained checkpoint: {cfg.checkpoint}")
    sam = build_sam2(
        cfg.sam_config,
        cfg.checkpoint,
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )
    return AutoSAM2(sam, cfg.prompt_tokens).to(device)
