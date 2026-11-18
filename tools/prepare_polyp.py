"""Convert the HSNet/PraNet-style polyp dataset into binary-mask manifests.

Reads images and masks, binarizes masks at a threshold, copies them into a
self-contained directory, and writes train/val CSVs with a deterministic split.
"""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def binarize(src, dst, threshold):
    with Image.open(src) as im:
        array = np.asarray(im.convert("L"))
    binary = (array >= threshold).astype(np.uint8) * 255
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"D:\HSNet-main\dataset\TrainDataset")
    parser.add_argument("--output", default=r"C:\Users\Administrator\repos\medsam-research\data\polyp")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="0 = all paired images")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="polyp-split-v1")
    parser.add_argument("--dataset-name", default="Kvasir")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.output)
    images = {p.stem: p for p in (source / "images").glob("*.png")}
    masks = {p.stem: p for p in (source / "masks").glob("*.png")}
    paired = sorted(set(images) & set(masks))
    if not paired:
        raise SystemExit("no paired images")
    # Deterministic, source-independent ordering by hashed id.
    ordered = sorted(paired, key=lambda s: hashlib.sha256(f"{args.seed}:{s}".encode()).hexdigest())
    if args.limit:
        ordered = ordered[: args.limit]
    val_count = max(1, round(len(ordered) * args.val_fraction)) if len(ordered) > 1 else 0
    val_ids = set(ordered[:val_count])
    splits = {"train": [s for s in ordered if s not in val_ids], "val": sorted(val_ids)}

    rows = {"train": [], "val": []}
    for split, ids in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "masks" / split).mkdir(parents=True, exist_ok=True)
        for stem in ids:
            shutil.copy2(images[stem], out / "images" / split / f"{stem}.png")
            binarize(masks[stem], out / "masks" / split / f"{stem}.png", args.threshold)
            rows[split].append({
                "id": stem,
                "image": f"images/{split}/{stem}.png",
                "mask": f"masks/{split}/{stem}.png",
                "dataset": "Kvasir" if stem.startswith("cju") else "ClinicDB",
                "group": "",
            })
        with (out / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "image", "mask", "dataset", "group"])
            writer.writeheader()
            writer.writerows(rows[split])
    print({"output": str(out), "train": len(rows["train"]), "val": len(rows["val"]),
           "threshold": args.threshold, "val_ids": splits["val"][:5]})


if __name__ == "__main__":
    main()
