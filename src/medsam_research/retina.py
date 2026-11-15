"""Import official retinal dataset layouts; never invent an official split."""

import csv
import gzip
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .data import audit_manifests, file_hash

CHASE_SHA256 = "bb750b73633492d700e56d91578ee3cdb7e4dbd08279e7465583e9efd26790b6"
STARE_URLS = {
    name: f"https://cecas.clemson.edu/~ahoover/stare/probing/{name}"
    for name in ("stare-images.tar", "labels-ah.tar", "labels-vk.tar")
}
STARE_IDS = {
    "im0001",
    "im0002",
    "im0003",
    "im0004",
    "im0005",
    "im0044",
    "im0077",
    "im0081",
    "im0082",
    "im0139",
    "im0162",
    "im0163",
    "im0235",
    "im0236",
    "im0239",
    "im0240",
    "im0255",
    "im0291",
    "im0319",
    "im0324",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".ppm", ".pgm"}
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024


def download_stare(output):
    """Fetch the original public files; no silent third-party mirror fallback."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for filename, url in STARE_URLS.items():
        target = output / filename
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(url, timeout=30) as response,
                temporary.open("xb") as stream,
            ):
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_TOTAL:
                        raise ValueError("Download exceeds dataset size limit")
                    stream.write(chunk)
            if not tarfile.is_tarfile(temporary):
                raise ValueError(f"{url} did not return a TAR archive")
            temporary.rename(target)
            receipts.append({"url": url, "file": filename, "sha256": file_hash(target)})
        finally:
            temporary.unlink(missing_ok=True)
    (output / "download.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return receipts


def _safe_name(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise ValueError(f"Unsafe archive member: {name}")
    return path.name


def _raw_files(sources):
    """Read bounded normal files without extracting untrusted archive paths."""
    files, receipts, total = {}, [], 0

    def accept(name, raw):
        nonlocal total
        name = _safe_name(name)
        logical = name[:-3] if name.lower().endswith(".gz") else name
        if Path(logical).suffix.lower() not in IMAGE_EXTENSIONS:
            return
        if len(raw) > MAX_FILE:
            raise ValueError(f"Oversized dataset member: {name}")
        if name.lower().endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE:
            raise ValueError(f"Oversized decompressed image: {name}")
        total += len(raw)
        if total > MAX_TOTAL:
            raise ValueError("Dataset exceeds expanded size limit")
        if logical in files:
            raise ValueError(f"Ambiguous duplicate filename across sources: {logical}")
        files[logical] = raw

    for source in map(Path, sources):
        source = source.resolve()
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Dataset symlink not supported: {path}")
                if path.is_file():
                    if path.stat().st_size > MAX_FILE:
                        raise ValueError(f"Oversized raw file: {path}")
                    accept(path.name, path.read_bytes())
            receipts.append({"path": str(source), "kind": "directory"})
        elif source.is_file():
            receipts.append({"path": str(source), "sha256": file_hash(source)})
            if zipfile.is_zipfile(source):
                with zipfile.ZipFile(source) as archive:
                    for member in archive.infolist():
                        _safe_name(member.filename)
                        if member.is_dir():
                            continue
                        if (member.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError("Archive symlinks are not supported")
                        if member.file_size > MAX_FILE:
                            raise ValueError("Oversized ZIP member")
                        accept(member.filename, archive.read(member))
            elif tarfile.is_tarfile(source):
                with tarfile.open(source) as archive:
                    for member in archive:
                        _safe_name(member.name)
                        if member.isdir():
                            continue
                        if not member.isfile() or member.size > MAX_FILE:
                            raise ValueError(
                                "TAR links, special files or oversized members rejected"
                            )
                        with archive.extractfile(member) as stream:
                            accept(member.name, stream.read(MAX_FILE + 1))
            else:
                raise ValueError(f"Expected a ZIP/TAR or extracted directory: {source}")
        else:
            raise FileNotFoundError(source)
    return files, receipts


def _find(files, pattern, required=True):
    matches = [name for name in files if re.fullmatch(pattern, name, flags=re.IGNORECASE)]
    if len(matches) > 1 or (required and not matches):
        raise ValueError(f"Expected one match for {pattern}, found {matches}")
    return matches[0] if matches else None


def _write_manifest(root, name, rows):
    with (root / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "image", "mask", "fov", "dataset", "group"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_sample(root, files, dataset, sid, image_name, labels, observer, group, fov=None):
    with Image.open(io.BytesIO(files[image_name])) as im:
        image = im.convert("RGB")
        image.save(root / "images" / f"{sid}.png")
        shape = image.size
    label_path = ""
    for index, filename in labels.items():
        if filename is None:
            continue
        with Image.open(io.BytesIO(files[filename])) as im:
            if im.size != shape:
                raise ValueError(f"Image/label size mismatch: {sid}, observer {index}")
            mask = np.asarray(im.convert("L")) > 0
        target = f"masks{index}/{sid}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(root / target)
        if index == observer:
            label_path = target
    fov_path = ""
    if fov:
        with Image.open(io.BytesIO(files[fov])) as im:
            if im.size != shape:
                raise ValueError(f"FOV size mismatch: {sid}")
            valid = np.asarray(im.convert("L")) > 0
        if not valid.any():
            raise ValueError(f"Empty FOV: {sid}")
        fov_path = f"fov/{sid}.png"
        Image.fromarray(valid.astype(np.uint8) * 255).save(root / fov_path)
    return {
        "id": sid,
        "image": f"images/{sid}.png",
        "mask": label_path,
        "fov": fov_path,
        "dataset": dataset,
        "group": group,
    }


def prepare_retina(dataset, sources, output, observer=1, seed=42):
    if observer not in (1, 2):
        raise ValueError("observer must be 1 or 2")
    if dataset not in {"CHASE_DB1", "DRIVE", "STARE"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not sources:
        if dataset != "CHASE_DB1":
            raise ValueError("Supply downloaded source archives/directories with --source")
        bundled = Path(__file__).resolve().parents[2] / "assets/datasets/CHASE_DB1/CHASEDB1.zip"
        if not bundled.exists() or file_hash(bundled) != CHASE_SHA256:
            raise ValueError("Bundled CHASE_DB1 archive missing/changed; use the Git checkout")
        sources = [bundled]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    files, receipts = _raw_files(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=".retina-", dir=output.parent))
    try:
        for name in ("images", "masks1", "masks2", "fov"):
            (root / name).mkdir()
        rows, official_train, official_test = [], [], []
        if dataset == "CHASE_DB1":
            for patient in range(1, 15):
                for eye in "LR":
                    sid = f"Image_{patient:02d}{eye}"
                    labels = {
                        i: _find(files, rf"{sid}_{'1st' if i == 1 else '2nd'}HO\.png")
                        for i in (1, 2)
                    }
                    rows.append(
                        _save_sample(
                            root,
                            files,
                            dataset,
                            sid,
                            _find(files, rf"{sid}\.jpg"),
                            labels,
                            observer,
                            f"child{patient:02d}",
                        )
                    )
            groups = sorted({r["group"] for r in rows})
            random.Random(seed).shuffle(groups)
            train_groups, val_groups = set(groups[:8]), set(groups[8:10])
            splits = {
                "train": [r for r in rows if r["group"] in train_groups],
                "val": [r for r in rows if r["group"] in val_groups],
                "test": [r for r in rows if r["group"] not in train_groups | val_groups],
            }
            protocol = "custom child-disjoint 8/2/4 subjects (16/4/8 images); NOT official"
        elif dataset == "STARE":
            for sid in sorted(STARE_IDS):
                labels = {
                    i: _find(files, rf"{sid}\.{who}\.(ppm|pgm|png)", required=i == observer)
                    for i, who in ((1, "ah"), (2, "vk"))
                }
                rows.append(
                    _save_sample(
                        root,
                        files,
                        dataset,
                        sid,
                        _find(files, rf"{sid}\.(ppm|png|jpg)"),
                        labels,
                        observer,
                        "",
                    )
                )
            splits = {"external": rows}
            protocol = "20-image external set only; no official train/test split asserted"
        else:
            for index in range(1, 41):
                split = "test" if index <= 20 else "training"
                sid = f"{index:02d}"
                labels = {
                    i: _find(
                        files,
                        rf"{sid}_manual{i}\.gif",
                        required=split == "training" and i == observer,
                    )
                    for i in (1, 2)
                }
                row = _save_sample(
                    root,
                    files,
                    dataset,
                    sid,
                    _find(files, rf"{sid}_{split}\.tif"),
                    labels,
                    observer,
                    "",
                    _find(files, rf"{sid}_{split}_mask\.gif"),
                )
                rows.append(row)
                (official_train if split == "training" else official_test).append(row)
            shuffled = official_train.copy()
            random.Random(seed).shuffle(shuffled)
            splits = {"train": shuffled[4:], "val": shuffled[:4]}
            labeled = [bool(r["mask"]) for r in official_test]
            if any(labeled) and not all(labeled):
                raise ValueError("Partially labeled DRIVE test set; supply all labels or none")
            splits["test" if all(labeled) else "test_inference"] = official_test
            protocol = (
                "official 20/20 train/test; custom 16/4 train/val within official training only"
            )
        _write_manifest(root, "all_labeled", [r for r in rows if r["mask"]])
        for name, split_rows in splits.items():
            _write_manifest(root, name, split_rows)
        audit_names = [name for name in splits if name != "test_inference"]
        audit_manifests([root / f"{name}.csv" for name in audit_names])
        fingerprint = hashlib.sha256()
        for name, raw in sorted(files.items()):
            fingerprint.update(name.encode())
            fingerprint.update(hashlib.sha256(raw).digest())
        provenance = {
            "dataset": dataset,
            "observer": observer,
            "seed": seed,
            "protocol": protocol,
            "sources": receipts,
            "source_members_sha256": fingerprint.hexdigest(),
            "sample_count": len(rows),
            "splits": {k: [r["id"] for r in v] for k, v in splits.items()},
            "groups": {r["id"]: r["group"] for r in rows},
            "fov_policy": "official supplied FOV"
            if dataset == "DRIVE"
            else "whole image; no synthetic FOV",
            "conversion": "RGB PNG without resizing; each observer mask separately binarized at >0",
        }
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        root.rename(output)
        return provenance
    except BaseException:
        shutil.rmtree(root)
        raise
