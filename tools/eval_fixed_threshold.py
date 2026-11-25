"""Fixed-threshold Dice/IoU from saved probability maps (paper convention).

Papers in the PraNet lineage report Dice after binarising at a fixed 0.5; this script
reproduces that convention so our numbers can sit next to published tables. It does not
replace the 256-threshold HSNet protocol, which stays in eval_hsnet_parallel.py.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

SETS = ("Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")


def dice_iou(pred, gt):
    tp = int((pred & gt).sum())
    denom = int(pred.sum()) + int(gt.sum())
    union = int((pred | gt).sum())
    if denom == 0:
        return 1.0, 1.0
    return 2.0 * tp / denom, (tp / union if union else 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", default="outputs")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--test-root", default="data/polyp_test_full")
    parser.add_argument("--manifest-suffix", default=".csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in SETS:
        pred_dir = Path(args.pred_root) / f"{args.prefix}{name}"
        preds = {r["id"]: r["prediction"] for r in
                 csv.DictReader(open(pred_dir / "predictions.csv", encoding="utf-8"))}
        gt_map = {r["id"]: r["mask"] for r in
                  csv.DictReader(open(Path(args.test_root) / f"{name}{args.manifest_suffix}",
                                      encoding="utf-8"))}
        rows = []
        for sample_id, filename in preds.items():
            prob = np.asarray(Image.open(pred_dir / filename).convert("L")).astype(np.float64) / 255.0
            gt = np.asarray(Image.open(Path(args.test_root) / gt_map[sample_id]).convert("L")) >= 128
            pred = prob >= args.threshold
            d, i = dice_iou(pred, gt)
            rows.append({"id": sample_id, "dice": d, "iou": i})
        summary[name] = {
            "n": len(rows),
            "dice": float(np.mean([r["dice"] for r in rows])),
            "iou": float(np.mean([r["iou"] for r in rows])),
            "threshold": args.threshold,
        }
        with (out / f"{name}_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "dice", "iou"])
            writer.writeheader()
            writer.writerows(rows)
        print(name, json.dumps(summary[name]))

    macro = {k: float(np.mean([summary[s][k] for s in SETS])) for k in ("dice", "iou")}
    summary["MACRO"] = macro
    with (out / "all_sets.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print("MACRO", json.dumps(macro))


if __name__ == "__main__":
    main()
