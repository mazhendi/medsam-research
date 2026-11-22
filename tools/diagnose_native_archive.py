"""Read the ORIGINAL result ZIP locally; export a small diagnostic CSV, no retraining.

Requires numpy and Pillow (already dependencies of medsam-research).
Example:
  python diagnose_archive.py native_sam2_baseline_2026-11-21.zip
No archive entries are extracted. Per-image best thresholds are diagnostic oracles,
NOT deployable thresholds and must never be reported as normal test performance.
"""

import argparse
import csv
import io
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def member_for(names, relative):
    relative = relative.replace("\\", "/").lstrip("./")
    candidates = [name for name in names if name == relative or name.endswith("/" + relative)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one ZIP member for {relative!r}; found {len(candidates)}")
    return candidates[0]


def diagnose(probability, mask):
    if probability.shape != mask.shape:
        raise ValueError(f"Native probability/mask shapes differ: {probability.shape}, {mask.shape}")
    gt = mask >= 128
    pred = probability >= 128
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    fg = int(gt.sum())
    npred = int(pred.sum())
    denominator = 2 * tp + fp + fn
    hist_fg = np.bincount(probability[gt], minlength=256)
    hist_bg = np.bincount(probability[~gt], minlength=256)
    cumulative_fg = hist_fg[::-1].cumsum()[::-1]
    cumulative_bg = hist_bg[::-1].cumsum()[::-1]
    denominators = cumulative_fg + cumulative_bg + fg
    dice = np.divide(2 * cumulative_fg, denominators,
                     out=np.ones(256, dtype=float), where=denominators > 0)
    best = int(dice.argmax())
    height, width = gt.shape
    return dict(
        width=width, height=height, gt_area_fraction=fg / gt.size,
        pred_area_fraction_05=npred / gt.size, gt_empty=fg == 0,
        empty_prediction_05=npred == 0,
        dice_05=2 * tp / denominator if denominator else 1.,
        iou_05=tp / (tp + fp + fn) if tp + fp + fn else 1.,
        precision_05=tp / npred if npred else (1. if fg == 0 else 0.),
        recall_05=tp / fg if fg else 1.,
        fp_pixels=fp, fn_pixels=fn,
        fp_per_gt=fp / fg if fg else None, fn_per_gt=fn / fg if fg else None,
        prob_max=float(probability.max()) / 255,
        mean_prob_fg=float(probability[gt].mean()) / 255 if fg else None,
        mean_prob_bg=float(probability[~gt].mean()) / 255 if (~gt).any() else None,
        oracle_best_dice=float(dice[best]), oracle_best_threshold=best / 255,
        image_diagonal=math.hypot(width, height),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", default="baseline_diagnosis.csv")
    parser.add_argument("--index-member", help="Exact ZIP member if it contains multiple index.csv files")
    args = parser.parse_args()
    output = Path(args.output)
    errors_path = output.with_name(output.stem + "_errors.csv")
    if output.exists() or errors_path.exists():
        raise FileExistsError("Choose a new output filename to preserve previous diagnostics")
    results, errors = [], []
    with ZipFile(args.archive) as archive:
        names = archive.namelist()
        if args.index_member:
            index = args.index_member
        else:
            candidates = [name for name in names if name.split("/")[-1] == "index.csv"]
            if len(candidates) != 1:
                raise ValueError("ZIP needs one index.csv; use --index-member to select it")
            index = candidates[0]
        rows = list(csv.DictReader(io.StringIO(archive.read(index).decode("utf-8-sig"))))
        for row in rows:
            try:
                with Image.open(io.BytesIO(archive.read(member_for(names, row["prediction"])))) as im:
                    probability = np.asarray(im.convert("L"))
                with Image.open(io.BytesIO(archive.read(member_for(names, row["mask"])))) as im:
                    mask = np.asarray(im.convert("L"))
                values = diagnose(probability, mask)
                # A zero/one GT encoding needs explicit conversion; do not silently call it empty.
                if mask.max() == 1:
                    raise ValueError("GT uses 0/1 encoding; convert explicitly before diagnosis")
                hd = float(row["hd95"])
                results.append(dict(set=row["set"], id=row["id"],
                                    baseline_meanDic=float(row["meanDic"]), hd95=hd,
                                    hd95_over_diagonal=hd / values["image_diagonal"], **values))
            except Exception as exc:
                errors.append(dict(set=row.get("set"), id=row.get("id"), error=str(exc)))
    for path, data in ((output, results), (errors_path, errors)):
        if data:
            with path.open("x", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    print(f"Processed {len(results)}/{len(rows)} images; errors={len(errors)}")
    if results:
        print(f"Upload {output}; no images or weights are included.")
    if errors:
        print(f"Also upload {errors_path}; do not interpret partial results as complete.")
    print("oracle_best_dice uses each test GT to choose a threshold: diagnosis only, not a valid score.")


if __name__ == "__main__":
    main()
