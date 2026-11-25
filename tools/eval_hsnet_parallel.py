"""Same metrics as tools/eval_hsnet_metrics.py, distributed over a process pool.

Kept as a separate driver so the validated per-image code path (HSNet author's
Fmeasure_calu/EnhancedMeasure/StructureMeasure/original_WFb, 256 thresholds,
optional soft maps) is untouched; only the loop is parallelised.
"""
import argparse
import csv
import json
import os
import sys
import types
from multiprocessing import Pool
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
METRIC_KEYS = ("meanDic", "maxDic", "meanIoU", "maxIoU", "meanF", "maxF",
               "meanEm", "maxEm", "meanSen", "maxSen", "meanSpe", "maxSpe",
               "wFm", "Sm", "mae")


def load_soft(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L")).astype(np.float64) / 255.0


def load_binary(path):
    with Image.open(path) as im:
        return (np.asarray(im.convert("L")) >= 128).astype(np.float64)


def one(task):
    sample_id, pred_path, gt_path, soft = task
    pred = load_soft(pred_path) if soft else load_binary(pred_path)
    gt = load_binary(gt_path)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {sample_id}: {pred.shape} vs {gt.shape}")
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
    return {
        "id": sample_id,
        "meanDic": float(np.mean(dice_s)), "maxDic": float(np.max(dice_s)),
        "meanIoU": float(np.mean(iou_s)), "maxIoU": float(np.max(iou_s)),
        "meanF": float(np.mean(f_s)), "maxF": float(np.max(f_s)),
        "meanEm": float(np.mean(e_s)), "maxEm": float(np.max(e_s)),
        "meanSen": float(np.mean(sen_s)), "maxSen": float(np.max(sen_s)),
        "meanSpe": float(np.mean(spe_s)), "maxSpe": float(np.max(spe_s)),
        "wFm": float(original_WFb(pred, gt)),
        "Sm": float(StructureMeasure(pred, gt)),
        "mae": float(np.mean(np.abs(pred - gt))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", default="outputs")
    parser.add_argument("--pred-prefix", default="polyp_pred_")
    parser.add_argument("--test-root", default="data/polyp_test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sets", nargs="+", required=True)
    parser.add_argument("--soft-maps", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--manifest-suffix", default="_dedup.csv")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    for name in args.sets:
        pred_dir = Path(args.pred_root) / f"{args.pred_prefix}{name}"
        rows = list(csv.DictReader(open(pred_dir / "predictions.csv", encoding="utf-8")))
        gt_manifest = {r["id"]: r["mask"] for r in
                       csv.DictReader(open(Path(args.test_root) / f"{name}{args.manifest_suffix}", encoding="utf-8"))}
        tasks = [(row["id"], str(pred_dir / row["prediction"]),
                  str(Path(args.test_root) / gt_manifest[row["id"]]), args.soft_maps) for row in rows]
        with Pool(args.workers) as pool:
            records = pool.map(one, tasks)
        summary = {key: float(np.mean([r[key] for r in records])) for key in METRIC_KEYS}
        summary["n"] = len(records)
        all_summary[name] = summary
        with (out / f"{name}.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        with (out / f"{name}_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(name, json.dumps(summary), flush=True)
    with (out / "all_sets.json").open("w", encoding="utf-8") as stream:
        json.dump(all_summary, stream, indent=2)


if __name__ == "__main__":
    main()
