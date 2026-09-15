"""Evaluate saved masks with explicit fixed-threshold segmentation metrics.

The repository's previous script printed PySODMetrics' *dynamic-threshold*
mean IoU as "mIoU", which is not the same metric as train.py's pooled IoU at
threshold 0.5.  This script reports both definitions separately when
--with_pysod is requested.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

from training_utils import BinaryCounts


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation masks")
    parser.add_argument("--dataset_name", default="dataset")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--pred_threshold", type=float, default=127.5)
    parser.add_argument("--gt_threshold", type=float, default=0.0)
    parser.add_argument("--allow_missing_pairs", action="store_true")
    parser.add_argument("--with_pysod", action="store_true")
    parser.add_argument("--output_json", default=None)
    return parser


def index_images(root: Path, label: str) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {root}")
    result: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in result:
            duplicates.setdefault(key, [result[key].name]).append(path.name)
        else:
            result[key] = path
    if duplicates:
        raise RuntimeError(f"Duplicate {label} stems: {duplicates}")
    return result


def load_gray(path: Path) -> np.ndarray:
    array = np.asarray(imageio.imread(path))
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {array.shape}: {path}")
    return array


def image_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float | int]:
    counts = BinaryCounts()
    counts.update(torch.from_numpy(pred), torch.from_numpy(gt))
    return counts.metrics()


def compute_pysod(pairs: Sequence[Tuple[Path, Path]]) -> Dict[str, float]:
    try:
        import py_sod_metrics
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "--with_pysod requires pysodmetrics. Install requirements.txt first."
        ) from exc

    fm = py_sod_metrics.Fmeasure()
    wfm = py_sod_metrics.WeightedFmeasure()
    sm = py_sod_metrics.Smeasure()
    em = py_sod_metrics.Emeasure()
    mae = py_sod_metrics.MAE()
    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    fmv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
            "dice": py_sod_metrics.DICEHandler(**sample_gray),
        }
    )
    for pred_path, gt_path in pairs:
        pred = load_gray(pred_path).astype(np.uint8)
        gt = load_gray(gt_path).astype(np.uint8)
        fm.step(pred=pred, gt=gt)
        wfm.step(pred=pred, gt=gt)
        sm.step(pred=pred, gt=gt)
        em.step(pred=pred, gt=gt)
        mae.step(pred=pred, gt=gt)
        fmv2.step(pred=pred, gt=gt)
    fm_result = fm.get_results()["fm"]
    em_result = em.get_results()["em"]
    fmv2_result = fmv2.get_results()
    return {
        "dynamic_threshold_mean_iou": float(fmv2_result["iou"]["dynamic"].mean()),
        "dynamic_threshold_mean_dice": float(fmv2_result["dice"]["dynamic"].mean()),
        "Smeasure": float(sm.get_results()["sm"]),
        "weighted_Fmeasure": float(wfm.get_results()["wfm"]),
        "adaptive_Fmeasure": float(fm_result["adp"]),
        "mean_Emeasure": float(em_result["curve"].mean()),
        "MAE": float(mae.get_results()["mae"]),
    }


def main(args: argparse.Namespace) -> None:
    pred_root = Path(args.pred_path)
    gt_root = Path(args.gt_path)
    predictions = index_images(pred_root, "prediction")
    ground_truth = index_images(gt_root, "ground truth")

    missing_predictions = sorted(set(ground_truth) - set(predictions))
    orphan_predictions = sorted(set(predictions) - set(ground_truth))
    if (missing_predictions or orphan_predictions) and not args.allow_missing_pairs:
        details = []
        if missing_predictions:
            details.append(
                f"missing predictions for {len(missing_predictions)} GT stems: "
                f"{missing_predictions[:10]}"
            )
        if orphan_predictions:
            details.append(
                f"orphan predictions for {len(orphan_predictions)} stems: "
                f"{orphan_predictions[:10]}"
            )
        raise FileNotFoundError("; ".join(details))
    common = sorted(set(predictions) & set(ground_truth))
    if not common:
        raise RuntimeError("No prediction/GT pairs found")

    pooled = BinaryCounts()
    per_image_rows: List[Dict[str, object]] = []
    positive_ious: List[float] = []
    all_ious: List[float] = []
    pairs: List[Tuple[Path, Path]] = []

    for key in common:
        pred_path, gt_path = predictions[key], ground_truth[key]
        pred_raw, gt_raw = load_gray(pred_path), load_gray(gt_path)
        if pred_raw.shape != gt_raw.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch for {key}: {pred_raw.shape} vs {gt_raw.shape}"
            )
        pred = pred_raw > args.pred_threshold
        gt = gt_raw > args.gt_threshold
        pooled.update(torch.from_numpy(pred), torch.from_numpy(gt))
        row = {"stem": key, **image_metrics(pred, gt), "gt_has_foreground": bool(gt.any())}
        per_image_rows.append(row)
        all_ious.append(float(row["iou"]))
        if gt.any():
            positive_ious.append(float(row["iou"]))
        pairs.append((pred_path, gt_path))

    result: Dict[str, object] = {
        "dataset_name": args.dataset_name,
        "pairs": len(common),
        "missing_predictions": missing_predictions,
        "orphan_predictions": orphan_predictions,
        "thresholds": {
            "prediction_raw_value_gt": args.pred_threshold,
            "ground_truth_raw_value_gt": args.gt_threshold,
        },
        "pooled_fixed_threshold": pooled.metrics(),
        "mean_per_image_iou_including_empty": float(np.mean(all_ious)),
        "mean_per_image_iou_positive_gt_only": (
            float(np.mean(positive_ious)) if positive_ious else None
        ),
        "positive_gt_images": len(positive_ious),
        "per_image": per_image_rows,
    }
    if args.with_pysod:
        result["pysod_dynamic_threshold"] = compute_pysod(pairs)

    output_json = Path(args.output_json) if args.output_json else pred_root / "evaluation.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)

    csv_path = output_json.with_suffix(".per_image.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)

    metrics = result["pooled_fixed_threshold"]
    print(args.dataset_name)
    print(f"Pairs:              {len(common)}")
    print(f"Pooled fixed IoU:   {metrics['iou']:.6f}")
    print(f"Pooled fixed Dice:  {metrics['dice']:.6f}")
    print(f"Precision:          {metrics['precision']:.6f}")
    print(f"Recall:             {metrics['recall']:.6f}")
    print(f"Mean image IoU:     {result['mean_per_image_iou_including_empty']:.6f}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main(build_parser().parse_args())
