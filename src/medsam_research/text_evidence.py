"""Pinned local BiomedCLIP bundle and spatially aligned frozen patch/text evidence."""

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import file_hash
from .engine import write_json

BIOMEDCLIP_REPO = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
BIOMEDCLIP_REVISION = '9f341de24bfb00180f1b847274256e9b65a3a32e'
TOKENIZER_REVISION = 'd673b8835373c6fa116d6d8006b33d48734e305d'


def prepare_biomedclip(output):
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install pip install -e ".[text]" first') from exc
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    # Download revisions by immutable commit, including the separate tokenizer/config.
    api = HfApi()
    revision = api.model_info(BIOMEDCLIP_REPO, revision=BIOMEDCLIP_REVISION).sha
    out.mkdir(parents=True)
    snapshot_download(BIOMEDCLIP_REPO, revision=revision, local_dir=str(out),
                      allow_patterns=['open_clip_config.json', 'open_clip_pytorch_model.bin'])
    config = json.loads((out/'open_clip_config.json').read_text())
    tokenizer_repo = config['model_cfg']['text_cfg']['hf_tokenizer_name']
    tokenizer_revision = api.model_info(tokenizer_repo, revision=TOKENIZER_REVISION).sha
    snapshot_download(tokenizer_repo, revision=tokenizer_revision, local_dir=str(out/'tokenizer'),
                      allow_patterns=['config.json', 'tokenizer*.json', 'special_tokens_map.json',
                                      'vocab.txt', 'added_tokens.json'])
    receipt = {'repo': BIOMEDCLIP_REPO, 'revision': revision, 'tokenizer_repo': tokenizer_repo,
               'tokenizer_revision': tokenizer_revision}
    receipt['files'] = {str(p.relative_to(out)).replace('\\', '/'): file_hash(p)
                        for p in sorted(out.rglob('*')) if p.is_file() and '.cache' not in p.parts}
    write_json(out/'bundle.json', receipt)
    return receipt


def bundle_identity(directory):
    root = Path(directory)
    receipt = json.loads((root/'bundle.json').read_text())
    for name, expected in receipt['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f'BiomedCLIP bundle changed: {name}')
    required = {'open_clip_config.json', 'open_clip_pytorch_model.bin', 'tokenizer/config.json'}
    if not required <= receipt['files'].keys():
        raise ValueError('Incomplete BiomedCLIP bundle')
    return receipt


class FrozenBiomedCLIP(nn.Module):
    """Project contextualized ViT patch tokens using the pretrained image projection.

    This is a patch-cosine baseline, NOT Simple-ViLMedSAM's information-bottleneck IPP.
    Input uses a square warp consistent with the SAM path, not a center crop.
    """

    def __init__(self, directory, device):
        super().__init__()
        try:
            import open_clip
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Install pip install -e ".[text]" first') from exc
        root = Path(directory).resolve()
        self.identity = bundle_identity(root)
        config = json.loads((root/'open_clip_config.json').read_text())
        model_cfg = copy.deepcopy(config['model_cfg'])
        model_cfg['text_cfg'].update(hf_model_name=str(root/'tokenizer'),
                                     hf_tokenizer_name=str(root/'tokenizer'),
                                     hf_model_pretrained=False)
        self.model = open_clip.CustomTextCLIP(**model_cfg)
        open_clip.load_checkpoint(self.model, str(root/'open_clip_pytorch_model.bin'),
                                  strict=True, weights_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True,
                                                       trust_remote_code=False)
        self.context_length = model_cfg['text_cfg']['context_length']
        self.text_dim = model_cfg['embed_dim']
        self.image_size = model_cfg['vision_cfg']['image_size']
        preprocess = config['preprocess_cfg']
        self.register_buffer('mean', torch.tensor(preprocess['mean'])[None, :, None, None])
        self.register_buffer('std', torch.tensor(preprocess['std'])[None, :, None, None])
        self.model.requires_grad_(False)
        self.to(device).eval()
        self._text_cache = {}
        trunk = self.model.visual.trunk
        if not hasattr(trunk, 'patch_embed') or not hasattr(trunk, 'num_prefix_tokens'):
            raise ValueError('Unsupported BiomedCLIP visual trunk; need ViT patch tokens')
        if hasattr(self.model.visual.head, 'pool'):
            raise ValueError('Attention pooling is not supported by this patch projection')

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_texts(self, prompts):
        device = self.mean.device
        for text in dict.fromkeys(prompts):
            if text not in self._text_cache:
                tokens = self.tokenizer(text, padding='max_length', truncation=False,
                                        max_length=self.context_length, return_tensors='pt')['input_ids']
                if tokens.shape[-1] > self.context_length:
                    raise ValueError(f'Prompt exceeds BiomedCLIP context ({self.context_length} tokens); shorten the description explicitly')
                self._text_cache[text] = self.model.encode_text(tokens.to(device), normalize=True)
        return torch.cat([self._text_cache[t] for t in prompts])

    @torch.no_grad()
    def forward(self, images, prompts):
        if len(images) != len(prompts):
            raise ValueError('One prompt per image is required')
        x = F.interpolate(images.float(), (self.image_size, self.image_size), mode='bilinear',
                          align_corners=False, antialias=True)
        x = (x-self.mean)/self.std
        trunk = self.model.visual.trunk
        tokens = trunk.forward_features(x)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError('Expected BxNxC ViT tokens')
        patches = tokens[:, trunk.num_prefix_tokens:]
        side = math.isqrt(patches.shape[1])
        if side*side != patches.shape[1]:
            raise ValueError('Non-square BiomedCLIP patch lattice')
        # Reuse pretrained norm/projection, never an untrained random text-image bridge.
        patches = trunk.fc_norm(patches)
        patches = F.normalize(self.model.visual.head(patches).float(), dim=-1)
        text = self.encode_texts(prompts).float()
        similarity = torch.einsum('bnd,bd->bn', patches, text)
        return similarity.reshape(len(images), 1, side, side), text


def read_prompt_map(path):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          or not v.strip() for k, v in value.items()):
        raise ValueError('Prompt map must be a JSON object mapping image IDs to nonempty text')
    return value


def resolve_prompts(ids, default, mapping, strict=False):
    if not isinstance(default, str) or not default.strip():
        raise ValueError('A nonempty default task prompt is required')
    missing = [sid for sid in ids if sid not in mapping]
    if strict and missing:
        raise ValueError(f'Missing per-image prompts: {missing[:5]}')
    return [mapping.get(sid, default) for sid in ids]
