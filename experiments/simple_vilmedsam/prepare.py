"""Fetch a pinned author checkout and apply narrowly scoped runtime repairs.

Upstream is kept outside tracked project sources. Original train/data scripts are
not used: audited manifest-based entry points live in run.py.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = '76d95d68fff037182cbdaf37fffb2d30bda2a9ec'
URL = 'https://github.com/qcc001/Simple-ViLMedSAM.git'
PATCHES = {
    'segment_anything/modeling/common.py': [
        ('from scripts.methods import *\n', ''),
        ('import cv2\n', ''),
    ],
    'segment_anything/predictor.py': [('Segment_Anything', 'segment_anything')],
    'sam_lora_image_encoder.py': [('Segment_Anything', 'segment_anything')],
    'scripts/utils.py': [('import matplotlib.pyplot as plt\n', '')],
    'segment_anything/build_sam.py': [('torch.load(f)', 'torch.load(f, weights_only=True)')],
}


def prepare(path):
    path = Path(path).resolve()
    if not path.exists():
        subprocess.run(['git', 'clone', URL, str(path)], check=True)
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True):
            raise ValueError('Refusing to change a modified upstream checkout')
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', COMMIT], check=True)
    changes = []
    for name, edits in PATCHES.items():
        original = subprocess.check_output(['git', '-C', str(path), 'show', f'{COMMIT}:{name}'])
        patched = original.decode()
        for old, new in edits:
            if old not in patched:
                raise ValueError(f'Pinned source no longer matches: {name}')
            patched = patched.replace(old, new)
        target = path/name
        current = target.read_bytes()
        if current not in (original, patched.encode()):
            raise ValueError(f'Refusing to overwrite other edits in {name}')
        target.write_text(patched)
        changes.append({'path': name, 'original_sha256': hashlib.sha256(original).hexdigest(),
                        'patched_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    untouched = {}
    for name in ('model/model.py', 'scripts/iba.py', 'segment_anything/modeling/transformer.py',
                 'segment_anything/modeling/image_encoder.py'):
        original = subprocess.check_output(['git', '-C', str(path), 'show', f'{COMMIT}:{name}'])
        if (path/name).read_bytes() != original:
            raise ValueError(f'Author algorithm source modified: {name}')
        untouched[name] = hashlib.sha256(original).hexdigest()
    receipt = {'url': URL, 'commit': COMMIT, 'patches': changes, 'unmodified_sha256': untouched,
               'scope': 'Import/runtime compatibility only; original train/data/map entry points remain unused.'}
    (path/'reproduction_patch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='vendor/Simple-ViLMedSAM')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source), indent=2))
