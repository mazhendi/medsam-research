"""Audited entry points around pinned author IPP/BID/LoRA components.

This is a reproduction harness, not a claim to reproduce published scores.
BiomedCLIP's timm API needs a tensor-output hook instead of upstream's HF
layer-replacement API; the author bottleneck class/objective are preserved.
"""
import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from prepare import COMMIT
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from medsam_research.data import audit_manifests, file_hash, load_fov, load_pair, read_manifest
from medsam_research.diagnostics import vessel_metrics
from medsam_research.engine import set_seed, write_csv, write_json
from medsam_research.metrics import binary_metrics
from medsam_research.text_evidence import FrozenBiomedCLIP


def author_modules(source):
    source = Path(source).resolve()
    receipt = json.loads((source/'reproduction_patch.json').read_text())
    if receipt['commit'] != COMMIT:
        raise ValueError('Run prepare.py for the pinned author version')
    for patch in receipt['patches']:
        if file_hash(source/patch['path']) != patch['patched_sha256']:
            raise ValueError('Author source changed after compatibility repair')
    for name, digest in receipt['unmodified_sha256'].items():
        if file_hash(source/name) != digest:
            raise ValueError(f'Author algorithm source changed: {name}')
    sys.path.insert(0, str(source))
    from model.model import Model
    from sam_lora_image_encoder import LoRA_Sam
    from scripts.iba import InformationBottleneck
    from segment_anything.build_sam import build_sam_vit_h
    return Model, LoRA_Sam, InformationBottleneck, build_sam_vit_h, receipt


def ib_map(backend, image, prompt, bottleneck_class, layer=8, beta=.1, var=2., steps=10, seed=42):
    """Same upstream stochastic bottleneck, Adam lr=1, beta*KL - cosine.

    Upstream stores estimated mean/std but does not use them in forward(). We
    retain that behavior explicitly; changing var currently has no effect.
    """
    if not 0 <= layer < len(backend.model.visual.trunk.blocks) or steps < 1 or beta <= 0 or var <= 0:
        raise ValueError('Invalid IB layer/steps/beta/var')
    set_seed(seed)
    width, height = image.size
    side = max(width, height)
    ox, oy = (side-width)//2, (side-height)//2
    square = Image.new('RGB', (side, side))
    square.paste(image, (ox, oy))
    x = torch.from_numpy(np.asarray(square.resize((224, 224), Image.Resampling.BICUBIC)).copy())
    x = x.permute(2, 0, 1)[None].to(backend.mean.device).float()/255
    x = (x-backend.mean)/backend.std
    block = backend.model.visual.trunk.blocks[layer]
    captured = []
    handle = block.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    try:
        with torch.no_grad():
            backend.model.encode_image(x)
    finally:
        handle.remove()
    feature = captured[0]
    bottleneck = bottleneck_class(np.zeros(feature.shape[1:]),
                                  np.full(feature.shape[1:], math.sqrt(var)), device=x.device)
    # Official HF layer returns a tuple; timm returns a tensor. Only this output
    # container is adapted. The author InformationBottleneck is used unchanged.
    handle = block.register_forward_hook(lambda module, args, output: bottleneck(output)[0])
    text = backend.encode_texts([prompt]).detach()
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=1.)
    history = []
    try:
        with torch.enable_grad():
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                visual = backend.model.encode_image(x, normalize=False)
                capacity = bottleneck.buffer_capacity
                compression = capacity.mean()
                fitting = F.cosine_similarity(text, visual, dim=-1).mean()
                loss = beta*compression-fitting
                loss.backward()
                norm = bottleneck.alpha.grad.norm().item()
                optimizer.step()
                history.append({'step': step+1, 'compression': compression.item(),
                                'cosine': fitting.item(), 'loss': loss.item(), 'alpha_grad_norm': norm})
    finally:
        handle.remove()
    # Match upstream: saliency from the last optimization forward, sum channel
    # capacities and discard CLS. Do not replace by direct patch/text cosine.
    raw = capacity.detach().mean(0).sum(-1)[1:]
    lattice = math.isqrt(raw.numel())
    raw = raw.reshape(1, 1, lattice, lattice)
    response = F.interpolate(raw, (224, 224), mode='bilinear', align_corners=False)[0, 0]
    span = (response.max()-response.min()).item()
    response = (response-response.min())/span if span > 1e-12 else torch.zeros_like(response)
    response = response.cpu().numpy()
    # cv2.INTER_NEAREST-compatible enlargement, then exact inverse of square pad.
    indices = np.minimum((np.arange(side)*224/side).astype(int), 223)
    native = response[indices[:, None], indices[None, :]][oy:oy+height, ox:ox+width]
    return native, {'history': history, 'raw_patch_capacity': raw[0, 0].cpu().tolist(),
                    'raw_response_span': span, 'flat': span <= 1e-12,
                    'native_std': float(native.std()), 'note': 'var retained but unused by author bottleneck forward'}


def generate_maps(args, modules):
    _, _, bottleneck_class, _, receipt = modules
    records = read_manifest(args.manifest, require_masks=False)
    backend = FrozenBiomedCLIP(args.biomedclip_dir, torch.device(args.device))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    index = {}
    for i, record in enumerate(records):
        started = time.perf_counter()
        image, _ = load_pair(record)
        response, diagnostics = ib_map(backend, image, args.prompt, bottleneck_class,
                                      args.layer, args.beta, args.var, args.steps, args.seed+i)
        filename = f'{i:06d}.png'
        Image.fromarray(np.round(response*255).astype(np.uint8)).save(out/filename)
        index[record.sample_id] = {'file': filename, 'image_sha256': file_hash(record.image),
                                   'map_sha256': file_hash(out/filename),
                                   'wall_seconds': time.perf_counter()-started}
        write_json(out/f'{i:06d}_diagnostics.json', diagnostics)
        print(f'IB map {record.sample_id}: std={response.std():.5f}', flush=True)
    write_json(out/'maps.json', {'format_version': 1, 'samples': index, 'prompt': args.prompt,
               'manifest_sha256': file_hash(args.manifest), 'biomedclip': backend.identity,
               'layer': args.layer, 'beta': args.beta, 'var': args.var, 'steps': args.steps,
               'seed': args.seed, 'author_source': receipt,
               'api_adaptation': 'OpenCLIP timm hook; official bottleneck and loss unchanged',
               'status': 'complete', 'warning': 'Unverified localization, not segmentation results'})


def padded_sample(record, map_root, size=1024, augment=False):
    root = Path(map_root)
    index = json.loads((root/'maps.json').read_text())
    if index.get('status') != 'complete':
        raise ValueError('Incomplete IB map export')
    entry = index['samples'].get(record.sample_id)
    if entry is None or entry['image_sha256'] != file_hash(record.image):
        raise ValueError(f'Missing or mismatched IB map for {record.sample_id}')
    path = (root/entry['file']).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != entry['map_sha256']:
        raise ValueError('Map identity/path check failed')
    image, mask = load_pair(record)
    if mask is None:
        raise ValueError('A training/evaluation mask is required')
    width, height = image.size
    shape = (int(height*size/max(height, width)+.5), int(width*size/max(height, width)+.5))
    rgb = np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)).copy()
    target = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST)).copy()
    with Image.open(path) as im:
        if im.size != image.size:
            raise ValueError('IB map dimensions differ from original image')
        attribution = np.asarray(im.convert('L').resize((shape[1], shape[0]), Image.Resampling.NEAREST)).copy()
    x = torch.zeros(1, 3, size, size)
    y = torch.zeros(1, 1, size, size)
    a = torch.zeros(1, 1, size, size)
    x[0, :, :shape[0], :shape[1]] = torch.from_numpy(rgb).permute(2, 0, 1)
    y[0, 0, :shape[0], :shape[1]] = torch.from_numpy(target)
    a[0, 0, :shape[0], :shape[1]] = torch.from_numpy(attribution)
    if augment:
        for axis in (-1, -2):
            if random.random() > .5:
                x, y, a = (v.flip(axis) for v in (x, y, a))
    return x, y, a, mask, shape


def dice_bce(logits, target):
    # MONAI DiceLoss(sigmoid=True,squared_pred=True,reduction='mean') defaults.
    p = logits.sigmoid()
    dims = tuple(range(2, p.ndim))
    dice = 1-(2*(p*target).sum(dims)+1e-5)/(p.square().sum(dims)+target.square().sum(dims)+1e-5)
    return F.binary_cross_entropy_with_logits(logits, target)+dice.mean()


def build_model(args, modules):
    Model, LoRA, _, build_sam, _ = modules
    if not Path(args.sam_checkpoint).is_file():
        raise FileNotFoundError('Download official SAM ViT-H checkpoint; SAM2 is not a substitute')
    sam = build_sam(str(Path(args.sam_checkpoint).resolve()))
    sam.requires_grad_(False)
    lora = LoRA(sam, 4).to(args.device)
    decoder = Model(embedding_dim=256, num_heads=8, mlp_dim=2048,
                    act=torch.nn.LeakyReLU).to(args.device)
    if getattr(args, 'activation_checkpointing', False):
        for block in sam.image_encoder.blocks:
            original = block.forward

            def wrapped(x, fn=original):
                if torch.is_grad_enabled():
                    return activation_checkpoint(fn, x, use_reentrant=False)
                return fn(x)

            block.forward = wrapped
    return sam, lora, decoder


def forward(sam, decoder, image, attribution):
    features = sam.image_encoder(sam.preprocess(image))
    attribution = F.interpolate(attribution, features.shape[-2:], mode='bilinear', align_corners=False)
    logits = decoder(features, attribution)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Author BID produced nonfinite output; inspect affinity masking before proceeding')
    return F.interpolate(logits, image.shape[-2:], mode='bilinear', align_corners=False)


@torch.no_grad()
def evaluate(sam, decoder, records, maps, device, output=None, map_mode='full'):
    sam.eval()
    decoder.eval()
    rows = []
    for i, record in enumerate(records):
        x, y, a, target, shape = padded_sample(record, maps)
        if map_mode == 'zero':
            a.zero_()
        elif map_mode == 'constant':
            a.fill_(float(a.mean()))
        logits = forward(sam, decoder, x.to(device), a.to(device))
        crop = logits[:, :, :shape[0], :shape[1]]
        native = F.interpolate(crop, target.shape, mode='bilinear', align_corners=False)[0, 0]
        prediction = (native.sigmoid() >= .5).cpu().numpy()
        valid = load_fov(record, target.shape)
        rows.append({'id': record.sample_id, **binary_metrics(prediction, target, valid=valid),
                     **vessel_metrics(prediction, target, valid),
                     'padded_grid_dice': binary_metrics((logits.sigmoid() >= .5)[0, 0].cpu().numpy(),
                                                       y[0, 0].numpy())['dice']})
        if output:
            Image.fromarray(prediction.astype(np.uint8)*255).save(Path(output)/f'{i:06d}_mask.png')
    return rows


def adaptation_state(lora):
    return {name: p.detach().cpu().clone() for name, p in lora.named_parameters() if p.requires_grad}


def restore_adapter(lora, saved):
    expected = {name: p for name, p in lora.named_parameters() if p.requires_grad}
    if set(expected) != set(saved):
        raise ValueError('LoRA checkpoint parameter names mismatch')
    with torch.no_grad():
        for name, value in saved.items():
            expected[name].copy_(value)


def train(args, modules):
    train_records, val_records = read_manifest(args.train_manifest), read_manifest(args.val_manifest)
    audit = audit_manifests([args.train_manifest, args.val_manifest], 0)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sam, lora, decoder = build_model(args, modules)
    sam_hash = file_hash(args.sam_checkpoint)
    if args.init:
        saved = torch.load(args.init, map_location='cpu', weights_only=True)
        if saved['provenance']['sam_sha256'] != sam_hash:
            raise ValueError('Initial checkpoint uses a different SAM')
        decoder.load_state_dict(saved['decoder'])
        restore_adapter(lora, saved['lora'])
    params = [p for p in lora.parameters() if p.requires_grad]
    cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimizer = cls([{'params': params, 'lr': args.lora_lr},
                     {'params': decoder.parameters(), 'lr': args.model_lr}], weight_decay=1e-4)
    total = args.epochs*math.ceil(len(train_records)/args.accumulation)
    if args.schedule == 'cosine':
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(.05*total), total)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)
    provenance = {'author_source': modules[-1], 'sam_sha256': sam_hash, 'audit': audit,
                  'train_maps_sha256': file_hash(Path(args.train_maps)/'maps.json'),
                  'val_maps_sha256': file_hash(Path(args.val_maps)/'maps.json'),
                  'args': vars(args), 'scope': 'Audited harness; no claim of published split/score reproduction',
                  'color': 'RGB (upstream loader uses cv2 BGR)', 'augmentation': 'paired horizontal/vertical flips',
                  'loss': 'BCE + squared-prediction Dice, smooth_nr/dr=1e-5',
                  'lora_trainable': sum(p.numel() for p in params),
                  'decoder_parameters': sum(p.numel() for p in decoder.parameters())}
    write_json(out/'provenance.json', provenance)
    best = -1.
    started = time.perf_counter()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs+1):
        sam.train()
        decoder.train()
        order = list(train_records)
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for i, record in enumerate(order):
            x, y, a, _, _ = padded_sample(record, args.train_maps, augment=True)
            if args.map_mode == 'zero':
                a.zero_()
            elif args.map_mode == 'constant':
                a.fill_(float(a.mean()))
            # Normalize final incomplete accumulation group by its actual size.
            group_start = (i//args.accumulation)*args.accumulation
            group_size = min(args.accumulation, len(order)-group_start)
            context = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else nullcontext()
            with context:
                logits = forward(sam, decoder, x.to(args.device), a.to(args.device))
                loss = dice_bce(logits.float(), y.to(args.device))
            (loss/group_size).backward()
            losses.append(loss.item())
            if (i+1) % args.accumulation == 0 or i+1 == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if args.schedule == 'cosine':
                    scheduler.step()
        if args.schedule == 'step':
            scheduler.step()
        rows = evaluate(sam, decoder, val_records, args.val_maps, args.device, map_mode=args.map_mode)
        score = float(np.mean([r['dice'] for r in rows]))
        write_csv(out/f'val_epoch_{epoch:03d}.csv', rows)
        entry = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_native_dice': score}
        with (out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(entry)+'\n')
        saved = {'kind': 'simple-vilmedsam-harness', 'epoch': epoch, 'decoder': decoder.state_dict(),
                 'lora': adaptation_state(lora), 'provenance': provenance}
        torch.save(saved, out/'last.pt')
        if score > best:
            best = score
            torch.save(saved, out/'best.pt')
        print(entry, flush=True)
    write_json(out/'summary.json', {'best_val_native_dice': best, 'wall_seconds': time.perf_counter()-started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None})


def smoke(args, modules):
    """Real ViT-H forward + author BID backward; no full LoRA training claim."""
    records = read_manifest(args.manifest)
    sam, lora, decoder = build_model(args, modules)
    sam.eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    x, y, a, _, _ = padded_sample(records[0], args.maps)
    with torch.no_grad():
        features = sam.image_encoder(sam.preprocess(x.to(args.device)))
    print('Official SAM ViT-H features ready', tuple(features.shape), flush=True)
    # Release encoder weights before decoder autograd on limited CPU RAM.
    del sam, lora
    import gc
    gc.collect()
    decoder.train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
    attribution = F.interpolate(a.to(args.device), features.shape[-2:], mode='bilinear', align_corners=False)
    before = decoder.output_upscaling1[-1].weight.detach().clone()
    logits = decoder(features.detach(), attribution)
    target = F.interpolate(y.to(args.device), logits.shape[-2:], mode='nearest')
    loss = dice_bce(logits, target)
    loss.backward()
    optimizer.step()
    change = (before-decoder.output_upscaling1[-1].weight).abs().max().item()
    write_json(out/'smoke.json', {'sam_sha256': file_hash(args.sam_checkpoint),
               'source_commit': COMMIT, 'image_id': records[0].sample_id,
               'features': list(features.shape), 'output': list(logits.shape), 'loss': loss.item(),
               'decoder_parameter_max_change': change, 'finite': bool(torch.isfinite(logits).all()),
               'scope': 'Real SAM ViT-H forward and author decoder optimizer step; not full training or accuracy reproduction'})
    torch.save({'decoder': decoder.state_dict()}, out/'decoder_smoke.pt')
    print(json.loads((out/'smoke.json').read_text()), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['maps', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--source', default='vendor/Simple-ViLMedSAM')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--biomedclip-dir', default='checkpoints/biomedclip')
    parser.add_argument('--prompt', default='polyp')
    parser.add_argument('--layer', type=int, default=8)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--var', type=float, default=2.)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--sam-checkpoint', default='checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--maps')
    parser.add_argument('--train-manifest')
    parser.add_argument('--val-manifest')
    parser.add_argument('--train-maps')
    parser.add_argument('--val-maps')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lora-lr', type=float, default=2e-4)
    parser.add_argument('--model-lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adamw')
    parser.add_argument('--schedule', choices=['cosine', 'step'], default='cosine')
    parser.add_argument('--map-mode', choices=['full', 'zero', 'constant'], default='full')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--amp', action='store_true', help='CUDA BF16; numerical deviation from FP32')
    parser.add_argument('--init', help='Weights-only initialization for adaptation/evaluation, not optimizer resume')
    args = parser.parse_args()
    if args.threads < 1 or args.accumulation < 1 or args.epochs < 1:
        parser.error('threads, accumulation and epochs must be positive')
    if args.amp and (not args.device.startswith('cuda') or not torch.cuda.is_bf16_supported()):
        parser.error('--amp requires CUDA BF16 support')
    required = {'maps': ['manifest'], 'smoke': ['manifest', 'maps'],
                'train': ['train_manifest', 'val_manifest', 'train_maps', 'val_maps'],
                'evaluate': ['manifest', 'maps', 'init']}[args.command]
    if any(getattr(args, key) is None for key in required):
        parser.error(f'{args.command} requires: {required}')
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    modules = author_modules(args.source)
    if args.command == 'maps':
        generate_maps(args, modules)
    elif args.command == 'smoke':
        smoke(args, modules)
    elif args.command == 'train':
        train(args, modules)
    else:
        sam, lora, decoder = build_model(args, modules)
        saved = torch.load(args.init, map_location='cpu', weights_only=True)
        if saved['provenance']['sam_sha256'] != file_hash(args.sam_checkpoint):
            raise ValueError('SAM identity mismatch')
        decoder.load_state_dict(saved['decoder'])
        restore_adapter(lora, saved['lora'])
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        rows = evaluate(sam, decoder, read_manifest(args.manifest), args.maps, args.device, out, args.map_mode)
        write_csv(out/'per_image.csv', rows)
        write_json(out/'summary.json', {'native_dice': float(np.mean([r['dice'] for r in rows])),
                   'checkpoint_sha256': file_hash(args.init), 'map_mode': args.map_mode,
                   'maps_sha256': file_hash(Path(args.maps)/'maps.json'),
                   'manifest_sha256': file_hash(args.manifest), 'author_source': modules[-1]})


if __name__ == '__main__':
    main()
