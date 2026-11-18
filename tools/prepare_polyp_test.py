"""Convert the five HSNet/PraNet-style polyp test sets into binary-mask manifests."""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SETS = ("Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")


def binarize(src, dst, threshold):
    with Image.open(src) as im:
        array = np.asarray(im.convert("L"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((array >= threshold) * 255).astype(np.uint8)).save(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"D:\HSNet-main\dataset\TestDataset")
    parser.add_argument("--output", default=r"C:\Users\Administrator\repos\medsam-research\data\polyp_test")
    parser.add_argument("--threshold", type=int, default=128)
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.output)
    for name in SETS:
        images = {p.stem: p for p in (source / name / "images").glob("*")}
        masks = {p.stem: p for p in (source / name / "masks").glob("*")}
        paired = sorted(set(images) & set(masks))
        if not paired:
            raise SystemExit(f"no paired files in {name}")
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "masks" / name).mkdir(parents=True, exist_ok=True)
        rows = []
        for stem in paired:
            shutil.copy2(images[stem], out / "images" / name / f"{stem}.png")
            binarize(masks[stem], out / "masks" / name / f"{stem}.png", args.threshold)
            rows.append({
                "id": stem,
                "image": f"images/{name}/{stem}.png",
                "mask": f"masks/{name}/{stem}.png",
                "dataset": name,
                "group": "",
            })
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "image", "mask", "dataset", "group"])
            writer.writeheader()
            writer.writerows(rows)
        print(name, len(rows))


if __name__ == "__main__":
    main()
