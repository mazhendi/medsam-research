"""Save native-resolution probability maps (8-bit) for HSNet-style metric sweeps."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from medsam_research.config import load_config
from medsam_research.data import image_tensor, load_pair, read_manifest
from medsam_research.engine import _restore, write_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = read_manifest(args.manifest, require_masks=False)
    model, device = _restore(cfg, args.checkpoint)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    mapping = []
    with torch.no_grad():
        for index, record in enumerate(records):
            image, _ = load_pair(replace(record, mask=None))
            tensor = image_tensor(image, model.image_size)[None].to(device)
            logits = model(tensor)
            logits = F.interpolate(logits.float(), size=(image.height, image.width),
                                   mode="bilinear", align_corners=False)
            prob = logits[0, 0].sigmoid().cpu().numpy()
            filename = f"{index:06d}.png"
            Image.fromarray(np.round(prob * 255).astype(np.uint8)).save(out / filename)
            mapping.append({"id": record.sample_id, "image": str(record.image),
                            "prediction": filename})
    write_csv(out / "predictions.csv", mapping)
    write_json(out / "provenance.json", {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "encoding": "8-bit probability map, value/255 approximates sigmoid(logit)",
        "threshold": cfg.training.threshold,
    })


if __name__ == "__main__":
    main()
