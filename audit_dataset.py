"""Preflight audit for paired RGB, binary mask and DTM tiles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from dataset import build_sample_records, read_dtm_masked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit RGB/mask/DTM data before training")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--dtm_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = audit all matched samples")
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--allow_missing_pairs", action="store_true")
    return parser


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def main(args: argparse.Namespace) -> None:
    records, pairing = build_sample_records(
        image_root=args.image_path,
        mask_root=args.mask_path,
        dtm_root=args.dtm_path,
        strict=not args.allow_missing_pairs,
    )
    selected = records if args.max_samples <= 0 else records[: args.max_samples]
    if not selected:
        raise RuntimeError("No matched samples to audit")

    rows: List[Dict[str, object]] = []
    mask_values = Counter()
    rgb_sizes = Counter()
    dtm_sizes = Counter()
    dtm_dtypes = Counter()
    dtm_crs = Counter()
    positive_tiles = 0
    foreground_pixels = 0
    total_mask_pixels = 0
    foreground_pixels_gt127 = 0
    shape_mismatches = 0
    all_nodata_tiles = 0
    constant_dtm_tiles = 0

    for index, record in enumerate(selected, start=1):
        with Image.open(record.image_path) as image:
            rgb_size = (image.height, image.width)
            rgb_mode = image.mode
        with Image.open(record.mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"))
        mask_size = tuple(mask.shape)
        unique, counts = np.unique(mask, return_counts=True)
        for value, count in zip(unique, counts):
            mask_values[int(value)] += int(count)

        foreground = mask > args.mask_threshold
        foreground_127 = mask > 127
        positive_tiles += int(foreground.any())
        foreground_pixels += int(foreground.sum())
        foreground_pixels_gt127 += int(foreground_127.sum())
        total_mask_pixels += int(mask.size)

        dtm, metadata = read_dtm_masked(record.dtm_path)
        finite = np.isfinite(dtm)
        finite_values = dtm[finite]
        if not finite.any():
            all_nodata_tiles += 1
            dtm_min = dtm_max = dtm_mean = dtm_std = None
        else:
            dtm_min = finite_or_none(finite_values.min())
            dtm_max = finite_or_none(finite_values.max())
            dtm_mean = finite_or_none(finite_values.mean())
            dtm_std = finite_or_none(finite_values.std())
            if float(finite_values.std()) <= 1e-6:
                constant_dtm_tiles += 1

        dtm_size = tuple(metadata["shape"])
        rgb_sizes[str(rgb_size)] += 1
        dtm_sizes[str(dtm_size)] += 1
        dtm_dtypes[str(metadata["dtype"])] += 1
        dtm_crs[str(metadata["crs"])] += 1
        shape_match = rgb_size == mask_size == dtm_size
        shape_mismatches += int(not shape_match)

        rows.append(
            {
                "stem": record.stem,
                "rgb_size": str(rgb_size),
                "rgb_mode": rgb_mode,
                "mask_size": str(mask_size),
                "mask_min": int(mask.min()),
                "mask_max": int(mask.max()),
                "mask_foreground_ratio_gt_config": float(foreground.mean()),
                "mask_foreground_ratio_gt127": float(foreground_127.mean()),
                "dtm_size": str(dtm_size),
                "dtm_dtype": metadata["dtype"],
                "dtm_nodata": metadata["nodata"],
                "dtm_nodata_is_nan": metadata["nodata_is_nan"],
                "dtm_finite_ratio": float(finite.mean()),
                "dtm_min": dtm_min,
                "dtm_max": dtm_max,
                "dtm_mean": dtm_mean,
                "dtm_std": dtm_std,
                "dtm_crs": metadata["crs"],
                "dtm_transform": str(metadata["transform"]),
                "all_shapes_equal": shape_match,
            }
        )
        if index % 100 == 0 or index == len(selected):
            print(f"[Audit] {index}/{len(selected)}")

    foreground_ratio = foreground_pixels / total_mask_pixels
    foreground_ratio_127 = foreground_pixels_gt127 / total_mask_pixels
    threshold_inflation = (
        foreground_ratio / foreground_ratio_127 if foreground_ratio_127 > 0 else None
    )
    summary = {
        "pairing": pairing.to_dict(),
        "audited_samples": len(selected),
        "positive_tiles": positive_tiles,
        "negative_tiles": len(selected) - positive_tiles,
        "foreground_ratio_at_config_threshold": foreground_ratio,
        "foreground_ratio_at_127": foreground_ratio_127,
        "foreground_inflation_factor_gt0_vs_gt127": threshold_inflation,
        "mask_unique_values": dict(sorted(mask_values.items())),
        "rgb_sizes": dict(rgb_sizes),
        "dtm_sizes": dict(dtm_sizes),
        "dtm_dtypes": dict(dtm_dtypes),
        "dtm_crs": dict(dtm_crs),
        "shape_mismatch_tiles": shape_mismatches,
        "all_nodata_dtm_tiles": all_nodata_tiles,
        "constant_dtm_tiles": constant_dtm_tiles,
        "important_limitation": (
            "Equal array dimensions do not prove RGB/DTM geospatial co-registration. "
            "PNG/JPEG RGB tiles do not carry a CRS/affine transform; verify alignment "
            "from the upstream tiling process or with visual overlays."
        ),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "samples": rows}, handle, indent=2, ensure_ascii=False, allow_nan=False)
    csv_path = output_json.with_suffix(".samples.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Saved] {output_json}")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
