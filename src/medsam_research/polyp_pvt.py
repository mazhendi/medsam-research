"""Load the pinned official Polyp-PVT architecture without implicit disk loads.

The upstream files remain in a separate checkout. Only the import namespace and
hard-coded pretrained loading are adapted in memory; all forward math is unchanged.
"""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import torch

UPSTREAM_COMMIT = "cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4"
SOURCE_HASHES = {
    "lib/pvt.py": "aa34ef3da7dcad9db95f757a3326e2b6b8839a14f7aadb34fab555bda4f0d1d9",
    "lib/pvtv2.py": "bad9a96bca9841679b5f7a5a97bf9bbc4239611fb4be6d0ba0783f64e4f9dc8c",
}


def build_polyp_pvt(upstream, encoder_checkpoint=None):
    root = Path(upstream).resolve()
    for filename, expected in SOURCE_HASHES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run tools/prepare_polyp_pvt.py first")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Polyp-PVT source differs from pinned revision: {filename}")
    namespace = "_medsam_official_polyp_pvt"
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(root / "lib")]
        sys.modules[namespace] = package
    name = namespace + ".pvt"
    if name not in sys.modules:
        backbone_name = namespace + ".pvtv2"
        spec = importlib.util.spec_from_file_location(backbone_name, root / "lib/pvtv2.py")
        backbone = importlib.util.module_from_spec(spec)
        sys.modules[backbone_name] = backbone
        try:
            spec.loader.exec_module(backbone)
        except ImportError as exc:
            sys.modules.pop(backbone_name, None)
            raise ImportError('Install Polyp-PVT dependencies: pip install -e ".[polyp-pvt]"') from exc
        source = (root / "lib/pvt.py").read_text()
        source = source.replace("from lib.pvtv2 import pvt_v2_b2",
                                f"from {backbone_name} import pvt_v2_b2")
        start = source.index("        path = './pretrained_pth/pvt_v2_b2.pth'")
        end = source.index("        self.Translayer2_0", start)
        source = source[:start] + source[end:]
        module = types.ModuleType(name)
        module.__file__ = str(root / "lib/pvt.py")
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        sys.modules[name] = module
    model = sys.modules[name].PolypPVT()
    if encoder_checkpoint is not None:
        state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Expected official PVTv2-B2 state dictionary")
        # Official ImageNet weights contain a classifier; segmentation backbone does not.
        state = {k: v for k, v in state.items() if k not in {"head.weight", "head.bias"}}
        model.backbone.load_state_dict(state, strict=True)
    return model
