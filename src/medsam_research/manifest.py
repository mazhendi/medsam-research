"""Pair local directories without inventing random patient/video splits."""

import csv
import os
from pathlib import Path

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _index(directory, suffix=""):
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(directory).with_suffix("")
        stem = relative.name
        if suffix:
            if not stem.endswith(suffix):
                raise ValueError(f"Mask filename does not end in {suffix!r}: {path}")
            stem = stem[: -len(suffix)]
        key = relative.with_name(stem).as_posix()
        if key in result:
            raise ValueError(f"Duplicate image stem: {result[key]} and {path}")
        result[key] = path
    if not result:
        raise ValueError(f"No supported images in {directory}")
    return result


def create_manifest(images, masks, output, dataset, mask_suffix=""):
    images, masks, output = Path(images).resolve(), Path(masks).resolve(), Path(output).resolve()
    if not images.is_dir() or not masks.is_dir():
        raise ValueError("Both image and mask directories must exist")
    image_map, mask_map = _index(images), _index(masks, mask_suffix)
    if set(image_map) != set(mask_map):
        raise ValueError(
            f"Unmatched images: {sorted(set(image_map) - set(mask_map))[:5]}; "
            f"unmatched masks: {sorted(set(mask_map) - set(image_map))[:5]}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image", "mask", "dataset", "group"])
        writer.writeheader()
        for key in image_map:
            writer.writerow(
                {
                    "id": key,
                    "image": os.path.relpath(image_map[key], output.parent),
                    "mask": os.path.relpath(mask_map[key], output.parent),
                    "dataset": dataset,
                    "group": "",
                }
            )
    return len(image_map)
