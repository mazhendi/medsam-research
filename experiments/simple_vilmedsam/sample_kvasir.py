"""Three real Kvasir-SEG pairs for component checks, NOT a benchmark split.

Official Simula download was unavailable during validation; this pinned plain
JPEG mirror is recorded explicitly. No claim of filename-level paper splits.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from medsam_research.data import file_hash

REPO = 'berkaytrhn/kvasir-seg'
REVISION = '2ff3756208198329d2a2ad6f8596f5bccdb52eae'
IDS = ['cju0qkwl35piu0993l0dewei2', 'cju0qoxqj9q6s0835b43399p4', 'cju0qx73cjw570799j4n5cjze']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/kvasir_reproduction_examples')
    parser.add_argument('--cache-dir', default='checkpoints/kvasir-smoke-cache')
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import hf_hub_download
    sources = {}
    for sid in IDS:
        source = {}
        for folder in ('images', 'masks'):
            path = hf_hub_download(REPO, f'Kvasir-SEG/{folder}/{sid}.jpg', repo_type='dataset', revision=REVISION,
                                    cache_dir=args.cache_dir)
            source[folder] = path
        sources[sid] = source
    root.mkdir(parents=True, exist_ok=False)
    (root/'images').mkdir()
    (root/'masks').mkdir()
    records, provenance = [], []
    for sid, source in sources.items():
        shutil.copyfile(source['images'], root/'images'/f'{sid}.jpg')
        with Image.open(source['masks']) as im:
            mask = (np.asarray(im.convert('L')) >= 128).astype(np.uint8)*255
        Image.fromarray(mask).save(root/'masks'/f'{sid}.png')
        records.append({'id': sid, 'image': f'images/{sid}.jpg', 'mask': f'masks/{sid}.png',
                        'dataset': 'Kvasir-SEG-smoke'})
        provenance.append({'id': sid, 'original_image_sha256': file_hash(source['images']),
                           'original_jpeg_mask_sha256': file_hash(source['masks']),
                           'binary_mask_sha256': file_hash(root/'masks'/f'{sid}.png')})
    with (root/'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (root/'provenance.json').write_text(json.dumps({'mirror': REPO, 'revision': REVISION,
        'samples': provenance, 'mask_conversion': 'JPEG grayscale >=128 -> binary PNG',
        'scope': 'Three fixed component-check examples, not official train/val/test splits'}, indent=2)+'\n')
    print(root/'samples.csv')


if __name__ == '__main__':
    main()
