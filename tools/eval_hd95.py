"""Compute HD95 at threshold 0.5 from saved probability maps and GT masks."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

SETS = ("Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")


def load_gt(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) >= 128


def load_prob(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L")).astype(np.float64) / 255.0


def surface(mask):
    structure = generate_binary_structure(mask.ndim, 1)
    return mask ^ binary_erosion(mask, structure=structure, border_value=0)


def hd95(pred, gt):
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        # Undefined when one side is empty; use the image diagonal as a conservative bound.
        return float(np.hypot(*gt.shape))
    pred_b, gt_b = surface(pred), surface(gt)
    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)
    distances = np.concatenate([dt_gt[pred_b], dt_pred[gt_b]])
    return float(np.percentile(distances, 95))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-root", default="outputs")
    parser.add_argument("--prefix", default="polyp_prob_")
    parser.add_argument("--test-root", default="data/polyp_test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--manifest-suffix", default="_dedup.csv")
    parser.add_argument("--output", default="reports/polyp_full_2026-11-18/hd95")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in SETS:
        pred_dir = Path(args.prob_root) / f"{args.prefix}{name}"
        rows = list(csv.DictReader(open(pred_dir / "predictions.csv", encoding="utf-8")))
        gt_map = {r["id"]: r["mask"] for r in
                  csv.DictReader(open(Path(args.test_root) / f"{name}{args.manifest_suffix}", encoding="utf-8"))}
        values, per_image = [], []
        for row in rows:
            prob = load_prob(pred_dir / row["prediction"])
            gt = load_gt(Path(args.test_root) / gt_map[row["id"]])
            value = hd95(prob >= args.threshold, gt)
            values.append(value)
            per_image.append({"id": row["id"], "hd95": value})
        summary[name] = {
            "n": len(values),
            "hd95_mean": float(np.mean(values)),
            "hd95_median": float(np.median(values)),
            "threshold": args.threshold,
            "units": "original-image pixels",
        }
        with (out / f"{name}_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "hd95"])
            writer.writeheader()
            writer.writerows(per_image)
        print(name, json.dumps(summary[name]))
    with (out / "all_sets.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)


if __name__ == "__main__":
    main()
