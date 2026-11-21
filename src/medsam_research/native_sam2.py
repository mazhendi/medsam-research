"""352-grid automatic SAM2 with independently switchable proposed modules.

RCSR mask assembly follows the native SAM2 MaskDecoder.predict_masks sequence;
native transformer, upsampling, hypernetwork and single-mask token are retained.
SAM2 source: Meta Platforms, Inc. and affiliates, Apache-2.0; see third_party.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import SAM2_COMMIT
from .model import upstream_commit
from .native_modules import RCSR, SBCA, SCDS
from .sam2_unet import Adapter


class NativeSAM2(nn.Module):
    """Inputs are ImageNet-normalized B×3×352×352, as in the new baseline dataset."""

    def __init__(self, sam, *, scds=False, sbca=False, rcsr=False, task_tokens=4):
        super().__init__()
        if sam.image_size != 352:
            raise ValueError("Build SAM2 with model.image_size=352")
        self.encoder = sam.image_encoder.trunk
        self.encoder.requires_grad_(False)
        self.encoder.blocks = nn.Sequential(*(Adapter(block) for block in self.encoder.blocks))
        self.neck = sam.image_encoder.neck
        self.decoder = sam.sam_mask_decoder
        self.prompt_encoder = sam.sam_prompt_encoder
        self.prompt_encoder.requires_grad_(False)
        self.decoder.dynamic_multimask_via_stability = False
        self.register_buffer("no_mem", sam.no_mem_embed.detach().reshape(1, -1, 1, 1).clone())
        self.add_no_mem = sam.directly_add_no_mem_embed
        self.task_tokens = nn.Parameter(torch.randn(1, task_tokens, 256) * .02)
        channels = list(reversed(self.encoder.channel_list))
        self.scds = nn.ModuleList(
            SCDS(c, tuple(channels[2:])) for c in channels[:2]
        ) if scds else nn.ModuleList()
        if sbca:
            attention = self.decoder.transformer.layers[0].cross_attn_token_to_image
            index = 2 if self.decoder.pred_obj_scores else 1
            self.decoder.transformer.layers[0].cross_attn_token_to_image = SBCA(attention, index)
        self.rcsr = RCSR() if rcsr else None
        self.module_flags = dict(scds=scds, sbca=sbca, rcsr=rcsr)

    @property
    def sbca(self):
        module = self.decoder.transformer.layers[0].cross_attn_token_to_image
        return module if isinstance(module, SBCA) else None

    def train(self, mode=True):
        super().train(mode)
        self.prompt_encoder.eval()
        return self

    def forward(self, images):
        if images.shape[1:] != (3, 352, 352):
            raise ValueError("NativeSAM2 expects normalized B×3×352×352 inputs")
        raw = self.encoder(images)
        if self.scds:
            raw = [self.scds[0](raw[0], raw[2:]), self.scds[1](raw[1], raw[2:]), *raw[2:]]
        features, _ = self.neck(raw)
        # Hiera-L 352: neck emits 88,44,22,11; native image encoder scalps 11.
        features = features[:3]
        embed = features[2] + self.no_mem if self.add_no_mem else features[2]
        high = [self.decoder.conv_s0(features[0]), self.decoder.conv_s1(features[1])]
        _, dense = self.prompt_encoder(points=None, boxes=None, masks=None)
        pe = self.prompt_encoder.get_dense_pe()
        if embed.shape[-2:] != pe.shape[-2:] or embed.shape[-2:] != dense.shape[-2:]:
            raise RuntimeError("Native prompt/image grid mismatch")
        sparse = self.task_tokens.expand(images.shape[0], -1, -1)
        if self.rcsr is None:
            masks = self.decoder(
                image_embeddings=embed, image_pe=pe, sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense.expand(images.shape[0], -1, -1, -1),
                multimask_output=False, repeat_image=False, high_res_features=high,
            )[0]
        else:
            masks = self._decode_with_rcsr(embed, pe, dense, sparse, high)
        return F.interpolate(masks, (352, 352), mode="bilinear", align_corners=False)

    def _decode_with_rcsr(self, embed, pe, dense, sparse, high):
        decoder = self.decoder
        output_tokens = [decoder.iou_token.weight, decoder.mask_tokens.weight]
        if decoder.pred_obj_scores:
            output_tokens.insert(0, decoder.obj_score_token.weight)
        tokens = torch.cat(output_tokens, 0)[None].expand(embed.shape[0], -1, -1)
        tokens = torch.cat((tokens, sparse), 1)
        hs, src = decoder.transformer(embed + dense, pe.expand_as(embed), tokens)
        # Single-mask token: obj-score (optional), IoU, then mask tokens.
        mask_index = 2 if decoder.pred_obj_scores else 1
        region = hs[:, mask_index]
        src = src.transpose(1, 2).reshape_as(embed)
        dc1, norm, activation1, dc2, activation2 = decoder.output_upscaling
        first = self.rcsr(dc1(src), high[1], region)
        first = activation1(norm(first + high[1]))
        feature = activation2(dc2(first) + high[0])
        weights = decoder.output_hypernetworks_mlps[0](region)
        return (weights[:, None] @ feature.flatten(2)).reshape(
            embed.shape[0], 1, *feature.shape[-2:]
        )

    def auxiliary_loss(self, target):
        if self.sbca is None or self.sbca.last_evidence is None:
            return target.new_zeros(())
        evidence = self.sbca.last_evidence
        occupancy = F.adaptive_avg_pool2d(target, evidence.shape[-2:])
        return F.binary_cross_entropy_with_logits(evidence, occupancy)


def build_native_model(cfg, device="cpu", *, allow_random=False):
    from sam2.build_sam import build_sam2

    if upstream_commit() != SAM2_COMMIT:
        raise RuntimeError(f"Install pinned official SAM2 commit {SAM2_COMMIT}")
    checkpoint = cfg.get("encoder_checkpoint")
    if not allow_random and (not checkpoint or not Path(checkpoint).is_file()):
        raise FileNotFoundError("Original SAM2 sam2_hiera_large.pt checkpoint is required")
    sam = build_sam2(
        cfg.get("sam_config", "configs/sam2/sam2_hiera_l.yaml"),
        None if allow_random else checkpoint, device="cpu", mode="eval",
        hydra_overrides_extra=["++model.image_size=352"], apply_postprocessing=False,
    )
    model = NativeSAM2(sam, **cfg.get("modules", {}), task_tokens=cfg.get("task_tokens", 4))
    return model.to(device)
