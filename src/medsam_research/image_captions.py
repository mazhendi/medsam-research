"""Image-only, local VLM descriptions; never reads masks, FOVs or model error images."""

import hashlib
import json
import random
import re
from pathlib import Path
from time import time_ns

import torch

from .data import file_hash, load_pair, read_manifest
from .engine import git_revision, write_json

CAPTION_MODELS = {
    'Qwen/Qwen2.5-VL-3B-Instruct': '66285546d2b821cf421d4f5eb2576359d3770cd3',
    'HuggingFaceTB/SmolVLM-500M-Instruct': '753c1752602b5e739af7a7fa6947a635051897a3',
}
FIELDS = ('appearance', 'visibility', 'regions', 'uncertainty')
TEMPLATE_VERSION = 1


def caption_instruction(task):
    if not isinstance(task, str) or not task.strip():
        raise ValueError('A nonempty segmentation task is required')
    return (
        f'Inspect only the supplied image. The segmentation target is: {task.strip()}. '
        'Describe visible image characteristics useful for segmentation, not a diagnosis. '
        'Return ONLY one JSON object with exactly four string fields: '
        '"appearance" (illumination, contrast, blur or artifacts), '
        '"visibility" (how clearly the target can actually be seen), '
        '"regions" (image-relative areas with clearly visible quality differences), '
        '"uncertainty" (what cannot be assessed reliably). '
        'Use concise English, at most 24 words per field. '
        'If a property cannot be assessed, say "not assessable". '
        'Do not invent hidden structures, exact target sizes, target counts, diagnoses, '
        'patient details or a segmentation mask. Do not claim unseen fine structures exist. '
        'Do not use filenames, annotations, or prior predictions. '
        'Image-relative left/right must not be presented as patient anatomical laterality.'
    )


def parse_description(raw):
    value = raw.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```':
            raise ValueError('Incomplete fenced JSON response')
        value = '\n'.join(lines[1:-1])
    try:
        description = json.loads(value)
    except json.JSONDecodeError as exc:
        # Some small VLMs emit all four labeled fields instead of JSON. Parse only
        # that complete, ordered structure; never invent or fill a missing field.
        matches = list(re.finditer(r'(?:^|[,;\n]\s*)(appearance|visibility|regions|uncertainty)\s*:',
                                   value, flags=re.IGNORECASE))
        if tuple(m.group(1).lower() for m in matches) != FIELDS:
            raise ValueError('Caption model did not return all four structured fields') from exc
        description = {m.group(1).lower(): value[m.end():matches[i+1].start()
                                               if i+1 < len(matches) else len(value)].strip(' ,;')
                       for i, m in enumerate(matches)}
    if not isinstance(description, dict) or set(description) != set(FIELDS):
        raise ValueError(f'Expected exactly these caption fields: {FIELDS}')
    result = {}
    for key in FIELDS:
        item = description[key]
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f'Caption field {key} must be a nonempty string')
        item = re.sub(r'\s+', ' ', item).strip()
        if len(item.split()) > 40 or len(item) > 400:
            raise ValueError(f'Caption field {key} is too long; do not silently truncate it')
        result[key] = item
    return result


def render_description(task, description):
    return (f'Segment the visible {task.strip()}. '
            f'Appearance: {description["appearance"]} '
            f'Target visibility: {description["visibility"]} '
            f'Regions: {description["regions"]} '
            f'Uncertainty: {description["uncertainty"]}')


class LocalCaptioner:
    """Standalone caption generation: never loads SAM or the segmentation model."""

    def __init__(self, model_id, revision, device, cache_dir, max_new_tokens=256):
        try:
            from huggingface_hub import HfApi
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError('Install pip install -e ".[text,captions]"') from exc
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu for debugging')
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        revision = revision or CAPTION_MODELS.get(model_id)
        if not revision:
            revision = HfApi().model_info(model_id).sha
        self.identity = {'model_id': model_id, 'revision': revision}
        processor_args = {'revision': revision, 'cache_dir': cache_dir, 'trust_remote_code': False}
        if model_id.startswith('Qwen/Qwen2.5-VL'):
            processor_args.update(min_pixels=56*56, max_pixels=392*392)
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_args)
        if 'SmolVLM' in model_id:
            self.processor.image_processor.do_image_splitting = False
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, trust_remote_code=False,
            torch_dtype=self.dtype, low_cpu_mem_usage=True,
            device_map={'': str(self.device)},
            attn_implementation='eager' if 'SmolVLM' in model_id else 'sdpa').eval()
        self.model.requires_grad_(False)
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def __call__(self, image, instruction):
        messages = [{'role': 'user', 'content': [{'type': 'image'},
                                                {'type': 'text', 'text': instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt')
        inputs = {key: value.to(device=self.device, dtype=self.dtype)
                  if torch.is_floating_point(value) else value.to(self.device)
                  for key, value in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                     do_sample=False, use_cache=True)
        return self.processor.batch_decode(output[:, inputs['input_ids'].shape[-1]:],
                                            skip_special_tokens=True)[0]


def _image_records(manifests):
    records, seen = [], {}
    for manifest in manifests:
        for record in read_manifest(manifest, require_masks=False):
            digest = file_hash(record.image)
            if record.sample_id in seen:
                if seen[record.sample_id] != digest:
                    raise ValueError(f'Image ID collision across manifests: {record.sample_id}')
                continue
            records.append(record)
            seen[record.sample_id] = digest
    return records, seen


def generate_image_prompts(manifests, output, task, model_id='Qwen/Qwen2.5-VL-3B-Instruct',
                           revision=None, device='cuda', cache_dir='checkpoints/caption-models',
                           max_new_tokens=256, resume=False):
    if max_new_tokens < 32:
        raise ValueError('max_new_tokens must be at least 32')
    instruction = caption_instruction(task)
    records, hashes = _image_records(manifests)
    out = Path(output)
    if out.exists() and not resume:
        raise FileExistsError(out)
    if resume and not (out/'provenance.json').exists():
        raise ValueError('Resume requires an existing caption provenance.json')
    model_revision = revision or CAPTION_MODELS.get(model_id)
    if not model_revision:
        from huggingface_hub import HfApi
        model_revision = HfApi().model_info(model_id).sha
    identity = {'model_id': model_id, 'revision': model_revision}
    protocol = {'template_version': TEMPLATE_VERSION, 'task': task.strip(),
                'caption_model': identity, 'instruction': instruction,
                'max_new_tokens': max_new_tokens, 'do_sample': False, 'image_sha256': hashes}
    previous = {}
    if resume:
        old = json.loads((out/'provenance.json').read_text())
        if old['protocol'] != protocol:
            raise ValueError('Caption protocol or source images changed; use a new output directory')
        cache_path = out/'records.jsonl'
        raw_cache = cache_path.read_bytes()
        lines = raw_cache.splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index != len(lines)-1 or line.endswith(b'\n'):
                    raise ValueError('Corrupted caption cache; only an interrupted final line can be recovered') from None
                (out/f'records.interrupted.{time_ns()}.jsonl').write_bytes(raw_cache)
                cache_path.write_bytes(b''.join(lines[:index]))
                break
            if item['status'] == 'ok':
                previous[item['id']] = item
    else:
        out.mkdir(parents=True)
        write_json(out/'provenance.json', {'protocol': protocol, 'project_commit': git_revision(),
                   'manifests': [{'path': str(Path(p).resolve()), 'sha256': file_hash(p)} for p in manifests],
                   'note': 'Local VLM uses original RGB images only; captions are unverified model observations.'})
        (out/'records.jsonl').touch()
    errors = []
    pending = [r for r in records if r.sample_id not in previous]
    captioner = (LocalCaptioner(model_id, model_revision, device, cache_dir, max_new_tokens)
                 if pending else None)
    if captioner is not None and captioner.identity != identity:
        raise ValueError('Captioner loaded a different model revision')
    with (out/'records.jsonl').open('a', encoding='utf-8') as stream:
        for record in records:
            if record.sample_id in previous:
                continue
            image, _ = load_pair(record)  # require_masks=False erased mask/FOV references
            raw = ''
            try:
                raw = captioner(image, instruction)
                description = parse_description(raw)
                prompt = render_description(task, description)
                item = {'id': record.sample_id, 'image_sha256': hashes[record.sample_id],
                        'decoded_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                        'raw': raw, 'description': description, 'prompt': prompt, 'status': 'ok',
                        'quality_flags': ['possible_instruction_echo'] if
                            'image-relative areas with clearly visible quality differences'
                            in description['regions'].lower() else []}
                previous[record.sample_id] = item
            except Exception as exc:
                item = {'id': record.sample_id, 'image_sha256': hashes[record.sample_id],
                        'raw': raw, 'status': 'error', 'error': str(exc)}
                errors.append(item)
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
            stream.flush()
            print(f'Caption {record.sample_id}: {item["status"]}', flush=True)
    prompts = {r.sample_id: previous[r.sample_id]['prompt'] for r in records if r.sample_id in previous}
    summary = {'images': len(records), 'completed': len(prompts), 'failed': len(errors),
               'unique_texts': len(set(prompts.values())),
               'instruction_echo_images': sum(bool(row.get('quality_flags')) for row in previous.values()),
               'all_identical': len(set(prompts.values())) == 1,
               'status': 'complete' if len(prompts) == len(records) else 'incomplete'}
    write_json(out/'summary.json', summary)
    if len(prompts) != len(records):
        # Never present a partial map as complete or fill failures with a generic task sentence.
        raise RuntimeError('Caption generation incomplete; inspect records.jsonl and resume. No prompts.json exported.')
    write_json(out/'prompts.json', prompts)
    write_json(out/'prompts.metadata.json', {'format_version': 1, 'task': task.strip(),
               'caption_model': identity, 'template_version': TEMPLATE_VERSION,
               'image_sha256': hashes, 'prompt_map_sha256': file_hash(out/'prompts.json'),
               'note': 'Generated from images only. Not expert annotations; review for unsupported claims.'})
    return summary


def caption_metadata(path):
    if path is None:
        return None
    path = Path(path)
    sidecar = path.with_name(path.stem+'.metadata.json')
    if not sidecar.exists():
        return None
    value = json.loads(sidecar.read_text())
    if value.get('format_version') != 1 or not isinstance(value.get('image_sha256'), dict):
        raise ValueError('Unsupported prompt metadata format')
    return value


def audit_image_prompts(records, task, mapping, metadata=None, strict=True):
    missing = [r.sample_id for r in records if r.sample_id not in mapping]
    if strict and missing:
        raise ValueError(f'Missing per-image prompts for {len(missing)} images: {missing[:5]}')
    checked = 0
    if metadata is not None:
        if metadata['task'].strip().casefold() != task.strip().casefold():
            raise ValueError('Caption bundle task differs from the segmentation task prompt')
        for record in records:
            expected = metadata['image_sha256'].get(record.sample_id)
            if expected is None or file_hash(record.image) != expected:
                raise ValueError(f'Caption/image identity mismatch: {record.sample_id}')
            checked += 1
    texts = [mapping.get(r.sample_id, task) for r in records]
    return {'samples': len(records), 'covered': len(records)-len(missing), 'unique_texts': len(set(texts)),
            'image_hashes_checked': checked, 'missing_ids': missing,
            'duplicate_fraction': 1-len(set(texts))/len(texts)}


def pair_image_prompts(records, mapping, pairing, seed):
    if pairing == 'matched':
        return dict(mapping)
    if pairing != 'shuffle':
        raise ValueError('Prompt pairing must be matched or shuffle')
    grouped = {}
    for record in records:
        grouped.setdefault(record.dataset, []).append(record.sample_id)
    result = dict(mapping)
    rng = random.Random(seed)
    for dataset in sorted(grouped):
        ids = grouped[dataset]
        if len(ids) < 2:
            raise ValueError('Shuffled control needs at least two images per dataset in each split')
        ids = sorted(ids)
        rng.shuffle(ids)
        for target, source in zip(ids, ids[1:]+ids[:1]):
            result[target] = mapping[source]
    return result
