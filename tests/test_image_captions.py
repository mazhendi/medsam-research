import csv
import json

import numpy as np
import pytest

from medsam_research import image_captions
from medsam_research.data import file_hash, read_manifest
from medsam_research.image_captions import (
    audit_image_prompts,
    generate_image_prompts,
    pair_image_prompts,
    parse_description,
)


class FixtureCaptioner:
    """Generation workflow fixture only; no claim about medical visual understanding."""

    calls = []
    fail_second = False

    def __init__(self, model_id, revision, device, cache_dir, max_new_tokens):
        self.identity = {'model_id': model_id, 'revision': revision}

    def __call__(self, image, instruction):
        self.calls.append((image.size, instruction))
        if self.fail_second and len(self.calls) == 2:
            raise RuntimeError('fixture generation failure')
        first = np.array(image)[0, 0].tolist()
        return json.dumps({'appearance': f'Fixture corner RGB values {first}.',
                           'visibility': 'Not assessable.', 'regions': 'Fixture only.',
                           'uncertainty': 'Not a clinical description.'})


def caption_manifest(tmp_path):
    from PIL import Image

    path = tmp_path/'input.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'image', 'mask', 'fov', 'dataset', 'group'])
        for i in range(2):
            Image.new('RGB', (12, 10), (10+40*i, 30, 50)).save(tmp_path/f'{i}.png')
            writer.writerow([str(i), f'{i}.png', 'missing_mask.png', 'missing_fov.png', 'fixture', str(i)])
    return path


def test_caption_generation_reads_images_only_and_exports_bound_map(tmp_path, monkeypatch):
    manifest = caption_manifest(tmp_path)
    FixtureCaptioner.calls, FixtureCaptioner.fail_second = [], False
    monkeypatch.setattr(image_captions, 'LocalCaptioner', FixtureCaptioner)
    out = tmp_path/'captions'
    summary = generate_image_prompts([manifest], out, 'retinal blood vessels', device='cpu')
    assert summary['status'] == 'complete' and summary['unique_texts'] == 2
    mapping = json.loads((out/'prompts.json').read_text())
    receipt = json.loads((out/'prompts.metadata.json').read_text())
    assert set(mapping) == {'0', '1'} and 'Fixture corner' in mapping['0']
    assert receipt['prompt_map_sha256'] == file_hash(out/'prompts.json')
    assert len(FixtureCaptioner.calls) == 2
    # A completed resume must not reload the VLM or regenerate anything.
    monkeypatch.setattr(image_captions, 'LocalCaptioner', lambda *a: pytest.fail('Unexpected model load'))
    assert generate_image_prompts([manifest], out, 'retinal blood vessels', device='cpu',
                                  resume=True) == summary
    with pytest.raises(ValueError, match='protocol'):
        generate_image_prompts([manifest], out, 'polyp', device='cpu', resume=True)


def test_failed_captions_are_not_replaced_by_uniform_text_and_can_resume(tmp_path, monkeypatch):
    manifest = caption_manifest(tmp_path)
    FixtureCaptioner.calls, FixtureCaptioner.fail_second = [], True
    monkeypatch.setattr(image_captions, 'LocalCaptioner', FixtureCaptioner)
    out = tmp_path/'captions'
    with pytest.raises(RuntimeError, match='incomplete'):
        generate_image_prompts([manifest], out, 'retinal blood vessels', device='cpu')
    assert not (out/'prompts.json').exists()
    assert json.loads((out/'summary.json').read_text())['completed'] == 1
    FixtureCaptioner.fail_second = False
    result = generate_image_prompts([manifest], out, 'retinal blood vessels', device='cpu', resume=True)
    assert result['completed'] == 2 and len(FixtureCaptioner.calls) == 3


def test_caption_parser_rejects_broken_or_unbounded_fields():
    with pytest.raises(ValueError):
        parse_description('Here is an image.')
    with pytest.raises(ValueError):
        parse_description('{"appearance":"bright"}')
    fields = dict.fromkeys(('appearance', 'visibility', 'regions', 'uncertainty'), 'visible')
    assert parse_description('```json\n'+json.dumps(fields)+'\n```') == fields
    fields['regions'] = 'word '*41
    with pytest.raises(ValueError, match='too long'):
        parse_description(json.dumps(fields))


def test_missing_prompts_wrong_image_and_deranged_pairing(tmp_path):
    manifest = caption_manifest(tmp_path)
    records = read_manifest(manifest, require_masks=False)
    with pytest.raises(ValueError, match='Missing per-image'):
        audit_image_prompts(records, 'vessels', {'0': 'clear'})
    receipt = {'task': 'vessels', 'image_sha256': {r.sample_id: file_hash(r.image) for r in records}}
    mapping = {'0': 'dark image', '1': 'bright image'}
    assert audit_image_prompts(records, 'vessels', mapping, receipt)['image_hashes_checked'] == 2
    receipt['image_sha256']['0'] = 'wrong'
    with pytest.raises(ValueError, match='identity mismatch'):
        audit_image_prompts(records, 'vessels', mapping, receipt)
    shuffled = pair_image_prompts(records, mapping, 'shuffle', 42)
    assert shuffled == {'0': mapping['1'], '1': mapping['0']}
    assert pair_image_prompts(records, mapping, 'shuffle', 42) == shuffled


def test_cross_manifest_id_collision_fails_before_loading_model(tmp_path, monkeypatch):
    first = caption_manifest(tmp_path)
    second = tmp_path/'second.csv'
    second.write_text('id,image\n0,1.png\n')
    monkeypatch.setattr(image_captions, 'LocalCaptioner', lambda *a: pytest.fail('Unexpected model load'))
    with pytest.raises(ValueError, match='ID collision'):
        generate_image_prompts([first, second], tmp_path/'out', 'vessels')


def test_complete_labeled_response_is_parsed_without_filling_fields():
    raw = 'appearance: bright, visibility: low, regions: upper image, uncertainty: fine details not assessable.'
    value = parse_description(raw)
    assert value['regions'] == 'upper image'
    with pytest.raises(ValueError):
        parse_description('appearance: bright, visibility: low')


def test_resume_recovers_only_interrupted_final_cache_line(tmp_path, monkeypatch):
    manifest = caption_manifest(tmp_path)
    FixtureCaptioner.calls, FixtureCaptioner.fail_second = [], False
    monkeypatch.setattr(image_captions, 'LocalCaptioner', FixtureCaptioner)
    out = tmp_path/'captions'
    generate_image_prompts([manifest], out, 'vessels', device='cpu')
    with (out/'records.jsonl').open('ab') as stream:
        stream.write(b'{"id": "broken')
    generate_image_prompts([manifest], out, 'vessels', device='cpu', resume=True)
    assert len(list(out.glob('records.interrupted.*.jsonl'))) == 1
    assert len(FixtureCaptioner.calls) == 2
