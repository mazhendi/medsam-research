"""Explicit image/mask pairing, native-grid targets, and split leakage checks."""

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    sample_id: str
    image: Path
    mask: Path | None
    dataset: str
    group: str
    fov: Path | None = None


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path, require_masks=True):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "image"} | ({"mask"} if require_masks else set())
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns {sorted(required)}")
        records, seen = [], set()
        for row in reader:
            sid = row["id"].strip()
            if not sid or sid in seen:
                raise ValueError(f"{path}: empty or duplicate id {sid!r}")
            seen.add(sid)
            image_value = row["image"].strip()
            mask_value = (row.get("mask") or "").strip() if require_masks else ""
            fov_value = (row.get("fov") or "").strip() if require_masks else ""
            if not image_value or (require_masks and not mask_value):
                raise ValueError(f"{path}: missing image/mask for {sid}")
            image = (path.parent / image_value).resolve()
            mask = (path.parent / mask_value).resolve() if mask_value else None
            fov = (path.parent / fov_value).resolve() if fov_value else None
            for f in (image, mask, fov):
                if f is not None and not f.is_file():
                    raise FileNotFoundError(f)
            dataset = (row.get("dataset") or "").strip()
            group = (row.get("group") or "").strip()
            if group and not dataset:
                raise ValueError("Rows with group IDs also require a dataset namespace")
            records.append(Record(sid, image, mask, dataset, group, fov))
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def load_pair(record, mask_threshold=0):
    with Image.open(record.image) as im:
        if im.mode not in {"RGB", "RGBA", "L", "P"}:
            raise ValueError(
                f"{record.image}: only 8-bit images are supported; "
                "convert high-bit-depth radiology data with an explicit intensity policy first"
            )
        image = im.convert("RGB").copy()
    mask = None
    if record.mask is not None:
        with Image.open(record.mask) as im:
            array = np.asarray(im)
        if array.ndim != 2 or array.dtype not in (np.uint8, np.bool_):
            raise ValueError(f"{record.mask}: expected an 8-bit single-channel binary mask")
        if array.shape != (image.height, image.width):
            raise ValueError(f"Image/mask size mismatch: {record.sample_id}")
        values = np.unique(array)
        if len(values) > 2 or (len(values) == 2 and values[0] != 0):
            raise ValueError(f"{record.mask}: not a binary mask; values={values[:10]}")
        mask = (array > mask_threshold).astype(np.uint8)
    return image, mask


def image_tensor(image, image_size):
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.array(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_fov(record, shape):
    """Use a supplied official FOV, otherwise explicitly evaluate the entire image."""
    if record.fov is None:
        return np.ones(shape, dtype=np.uint8)
    with Image.open(record.fov) as im:
        array = np.asarray(im)
    if array.ndim != 2 or array.shape != shape or array.dtype not in (np.uint8, np.bool_):
        raise ValueError(f"Invalid FOV shape/type: {record.fov}")
    if len(np.unique(array)) > 2:
        raise ValueError(f"FOV is not binary: {record.fov}")
    valid = (array > 0).astype(np.uint8)
    if not valid.any():
        raise ValueError(f"Empty FOV: {record.fov}")
    return valid


class BinaryDataset(Dataset):
    def __init__(self, records, image_size, mask_threshold=0, augmentation="none"):
        if augmentation not in ("none", "light"):
            raise ValueError("augmentation must be 'none' or 'light'")
        self.records = records
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.augmentation = augmentation

    def _augment(self, image, mask, valid):
        """Geometric transforms apply to image, mask and FOV together; photometric image-only."""
        from PIL import ImageEnhance

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask[:, ::-1].copy()
            valid = valid[:, ::-1].copy()
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask[::-1, :].copy()
            valid = valid[::-1, :].copy()
        turns = random.choice((0, 0, 1, 2, 3))
        if turns:
            image = image.transpose({1: Image.Transpose.ROTATE_90,
                                     2: Image.Transpose.ROTATE_180,
                                     3: Image.Transpose.ROTATE_270}[turns])
            mask = np.rot90(mask, turns).copy()
            valid = np.rot90(valid, turns).copy()
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return image, mask, valid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, mask = load_pair(record, self.mask_threshold)
        if mask is None:
            raise ValueError("Training/evaluation requires masks")
        if self.augmentation == "light":
            valid = load_fov(record, mask.shape)
            image, mask, valid = self._augment(image, mask, valid)
        else:
            valid = load_fov(record, mask.shape)
        small = Image.fromarray(mask).resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        small_valid = Image.fromarray(valid).resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        return {
            "image": image_tensor(image, self.image_size),
            "mask": torch.from_numpy(np.array(small, dtype=np.float32))[None],
            "native_mask": mask,
            "valid": torch.from_numpy(np.array(small_valid, dtype=np.float32))[None],
            "native_valid": valid,
            "fov_used": record.fov is not None,
            "id": record.sample_id,
        }


def collate_samples(samples):
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
        "valid": torch.stack([s["valid"] for s in samples]),
        "native_mask": [s["native_mask"] for s in samples],
        "native_valid": [s["native_valid"] for s in samples],
        "fov_used": [s["fov_used"] for s in samples],
        "id": [s["id"] for s in samples],
    }


def audit_manifests(paths, mask_threshold=0):
    """Reject exact decoded-image duplicates and known group overlap, including within splits.

    This is NOT a near-duplicate detector. Patient/video groups require user metadata.
    """
    seen_images, group_splits, reports = {}, {}, []
    for split_index, path in enumerate(paths):
        records = read_manifest(path)
        empty, ungrouped, fov_count = 0, 0, 0
        content_hash = hashlib.sha256()
        for record in records:
            image, mask = load_pair(record, mask_threshold)
            valid = load_fov(record, mask.shape)
            fov_count += int(record.fov is not None)
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest in seen_images:
                raise ValueError(f"Duplicate image: {record.image} and {seen_images[digest]}")
            seen_images[digest] = str(record.image)
            content_hash.update(record.sample_id.encode())
            content_hash.update(digest.encode())
            content_hash.update(mask.tobytes())
            content_hash.update(valid.tobytes())
            if record.group:
                key = (record.dataset, record.group)
                if key in group_splits and group_splits[key] != split_index:
                    raise ValueError(f"Patient/video group occurs across splits: {key}")
                group_splits[key] = split_index
            else:
                ungrouped += 1
            empty += int(not (mask & valid).any())
        reports.append(
            {
                "manifest": str(Path(path).resolve()),
                "sha256": file_hash(path),
                "decoded_data_sha256": content_hash.hexdigest(),
                "samples": len(records),
                "empty_masks": empty,
                "samples_without_group": ungrouped,
                "samples_with_fov": fov_count,
            }
        )
    return reports
