"""Compute HSNet's polyp metrics on project predictions using the author's own functions.

Imports D:\\HSNet-main\\utils\\eval_functions.py unchanged; older scipy module paths are
shimmed because scipy >= 1.10 removed scipy.ndimage.morphology/filters.
"""
import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

for old, attr in (("scipy.ndimage.morphology", "distance_transform_edt"),
                  ("scipy.ndimage.filters", "convolve")):
    module = types.ModuleType(old)
    setattr(module, attr, getattr(ndi, attr))
    sys.modules[old] = module

HSNET_ROOT = Path(r"D:\HSNet-main")
sys.path.insert(0, str(HSNET_ROOT))
from utils.eval_functions import (  # noqa: E402
    EnhancedMeasure,
    Fmeasure_calu,
    StructureMeasure,
    original_WFb,
)

THRESHOLDS = np.linspace(1, 0, 256)
SETS = ("Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")


def load_binary(path):
    with Image.open(path) as im:
        return (np.asarray(im.convert("L")) >= 128).astype(np.float64)


def load_soft(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L")).astype(np.float64) / 255.0


def evaluate_set(pred_dir, test_root, name, soft=False):
    pred_dir = Path(pred_dir)
    rows = list(csv.DictReader(open(pred_dir / "predictions.csv", encoding="utf-8")))
    gt_manifest = {r["id"]: r["mask"] for r in
                   csv.DictReader(open(Path(test_root) / f"{name}_dedup.csv", encoding="utf-8"))}
    metrics = {key: [] for key in (
        "meanDic", "maxDic", "meanIoU", "maxIoU", "meanF", "maxF", "meanEm", "maxEm",
        "meanSen", "maxSen", "meanSpe", "maxSpe", "wFm", "Sm", "mae")}
    per_image = []
    for row in rows:
        pred = load_soft(pred_dir / row["prediction"]) if soft else load_binary(pred_dir / row["prediction"])
        gt = load_binary(Path(test_root) / gt_manifest[row["id"]])
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch {row['id']}: {pred.shape} vs {gt.shape}")
        dice_s, iou_s, f_s, e_s, sen_s, spe_s = [], [], [], [], [], []
        for t in THRESHOLDS:
            precision, recall, specificity, dice, fmeasure, iou = Fmeasure_calu(pred, gt, t)
            enhanced = EnhancedMeasure((pred >= t).astype(np.float64), gt)
            dice_s.append(dice)
            iou_s.append(iou)
            f_s.append(fmeasure)
            e_s.append(enhanced)
            sen_s.append(recall)
            spe_s.append(specificity)
        record = {
            "meanDic": float(np.mean(dice_s)), "maxDic": float(np.max(dice_s)),
            "meanIoU": float(np.mean(iou_s)), "maxIoU": float(np.max(iou_s)),
            "meanF": float(np.mean(f_s)), "maxF": float(np.max(f_s)),
            "meanEm": float(np.mean(e_s)), "maxEm": float(np.max(e_s)),
            "meanSen": float(np.mean(sen_s)), "maxSen": float(np.max(sen_s)),
            "meanSpe": float(np.mean(spe_s)), "maxSpe": float(np.max(spe_s)),
            "wFm": float(original_WFb(pred, gt)), "Sm": float(StructureMeasure(pred, gt)),
            "mae": float(np.mean(np.abs(pred - gt))),
        }
        record["id"] = row["id"]
        per_image.append(record)
        for key in metrics:
            metrics[key].append(record[key])
    summary = {key: float(np.mean(values)) for key, values in metrics.items()}
    summary["n"] = len(per_image)
    return summary, per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", default="outputs")
    parser.add_argument("--pred-prefix", default="polyp_pred_")
    parser.add_argument("--test-root", default="data/polyp_test")
    parser.add_argument("--output", default="reports/polyp_full_2026-11-18/hsnet_metrics")
    parser.add_argument("--soft-maps", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    for name in SETS:
        summary, per_image = evaluate_set(Path(args.pred_root) / f"{args.pred_prefix}{name}",
                                          args.test_root, name, soft=args.soft_maps)
        all_summary[name] = summary
        with (out / f"{name}.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        with (out / f"{name}_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
            writer.writeheader()
            writer.writerows(per_image)
        print(name, json.dumps(summary))
    with (out / "all_sets.json").open("w", encoding="utf-8") as stream:
        json.dump(all_summary, stream, indent=2)


if __name__ == "__main__":
    main()
