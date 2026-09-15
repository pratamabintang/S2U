"""Inference for SAM2-UNet RGB+DTM with checkpoint-config validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from dataset import TestDataset
from training_utils import BinaryCounts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test SAM2-UNet with RGB + DTM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hiera_path", required=True)
    parser.add_argument("--model_cfg", default=None)
    parser.add_argument("--test_image_path", required=True)
    parser.add_argument("--test_dtm_path", required=True)
    parser.add_argument("--test_gt_path", default=None, help="Optional; inference does not require GT")
    parser.add_argument("--save_path", required=True)

    parser.add_argument("--testsize", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=None,
        help="Ground-truth raw-value threshold. Defaults to the checkpoint mask threshold.",
    )
    parser.add_argument("--dtm_scale", type=float, default=None)
    parser.add_argument(
        "--dtm_availability",
        type=float,
        default=None,
        help=(
            "Defaults to the checkpoint training availability (or 1.0 for a legacy "
            "checkpoint). A different value is an ablation and requires "
            "--allow_preprocessing_override."
        ),
    )
    parser.add_argument("--availability_seed", type=int, default=None)
    parser.add_argument(
        "--dtm_norm",
        choices=("per_tile_zscore", "uint16_01", "global_zscore", "none"),
        default=None,
    )
    parser.add_argument("--dtm_divisor", type=float, default=None)
    parser.add_argument("--dtm_global_mean", type=float, default=None)
    parser.add_argument("--dtm_global_std", type=float, default=None)
    parser.add_argument("--dtm_missing_value", type=float, default=None)
    parser.add_argument("--allow_preprocessing_override", action="store_true")
    parser.add_argument("--allow_missing_pairs", action="store_true")
    parser.add_argument("--save_probability_uint16", action="store_true")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--erode_kernel",
        type=int,
        default=0,
        help=(
            "Kernel size (pixels) for morphological post-processing on the binary mask, "
            "applied after thresholding. 0 disables this entirely (default; identical to "
            "not passing this flag at all)."
        ),
    )
    parser.add_argument(
        "--erode_iter",
        type=int,
        default=1,
        help="Number of iterations for --erode_kernel's morphological operation.",
    )
    parser.add_argument(
        "--morph_op",
        choices=("erode", "open", "close", "dilate"),
        default="open",
        help=(
            "Which morphological operation --erode_kernel triggers. 'open' (erode then "
            "dilate) removes small isolated false-positive blobs while restoring crack "
            "pixels that are AT LEAST as wide as the kernel. It does NOT rescue structures "
            "thinner than the kernel: erosion can reduce a hairline crack to nothing, and "
            "dilating nothing still gives nothing. Inspect a few predicted masks first to "
            "see how thick your model's crack predictions actually are before choosing a "
            "kernel size."
        ),
    )
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=0,
        help=(
            "Remove connected white components smaller than this many pixels, applied "
            "BEFORE --erode_kernel. Unlike erosion, this does not shrink or thin surviving "
            "components at all -- a large crack blob is kept pixel-for-pixel identical, "
            "including thin necks connecting parts of it. Use this instead of erosion when "
            "you only want to drop small isolated false-positive specks. 0 disables this "
            "(default)."
        ),
    )
    return parser


def torch_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _nested(mapping: Mapping[str, Any], *keys: str, default=None):
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _resolve(
    name: str,
    explicit,
    checkpoint_value,
    fallback,
    allow_override: bool,
):
    if explicit is None:
        return checkpoint_value if checkpoint_value is not None else fallback
    if (
        checkpoint_value is not None
        and explicit != checkpoint_value
        and not allow_override
    ):
        raise ValueError(
            f"--{name}={explicit!r} conflicts with checkpoint value {checkpoint_value!r}. "
            "Use --allow_preprocessing_override only for an intentional ablation."
        )
    return explicit


def resolve_prediction_threshold(
    explicit: Optional[float], checkpoint_value: Optional[float], fallback: float = 0.5
) -> float:
    """Resolve an operating threshold without treating valid ``0.0`` as missing."""

    if explicit is not None:
        return float(explicit)
    if checkpoint_value is not None:
        return float(checkpoint_value)
    return float(fallback)


def remove_small_components(binary_uint8: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` pixels.

    Unlike erosion, surviving components are copied verbatim -- no shrinkage,
    no thinning, no risk of severing a thin neck between two blobby regions.
    """

    if min_area <= 0:
        return binary_uint8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_uint8, connectivity=8)
    output = np.zeros_like(binary_uint8)
    for label in range(1, num_labels):  # label 0 is background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 255
    return output


def apply_morphology(binary_uint8: np.ndarray, kernel_size: int, iterations: int, op: str) -> np.ndarray:
    """Post-process a 0/255 binary mask. No-op when kernel_size <= 0."""

    if kernel_size <= 0:
        return binary_uint8

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if op == "erode":
        return cv2.erode(binary_uint8, kernel, iterations=iterations)
    if op == "dilate":
        return cv2.dilate(binary_uint8, kernel, iterations=iterations)
    if op == "open":
        return cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if op == "close":
        return cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    raise ValueError(f"Unknown --morph_op {op!r}")


def validate_paths(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "hiera_path"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f"--{name} not found: {getattr(args, name)}")
    for name in ("test_image_path", "test_dtm_path"):
        if not Path(getattr(args, name)).is_dir():
            raise FileNotFoundError(f"--{name} not found: {getattr(args, name)}")
    if args.test_gt_path and not Path(args.test_gt_path).is_dir():
        raise FileNotFoundError(f"--test_gt_path not found: {args.test_gt_path}")
    if args.dtm_availability is not None and not 0 <= args.dtm_availability <= 1:
        raise ValueError("--dtm_availability must be in [0,1]")
    if args.erode_kernel < 0:
        raise ValueError(f"--erode_kernel must be >= 0 (0 = disabled), got {args.erode_kernel}")
    if args.erode_iter < 1:
        raise ValueError(f"--erode_iter must be >= 1, got {args.erode_iter}")
    if args.min_component_area < 0:
        raise ValueError(
            f"--min_component_area must be >= 0 (0 = disabled), got {args.min_component_area}"
        )


def main(args: argparse.Namespace) -> None:
    validate_paths(args)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch_load(checkpoint_path)
    state_dict = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint

    checkpoint_train_pre = _nested(checkpoint, "data_config", "train_preprocessing", default={}) or {}
    checkpoint_run = checkpoint.get("run_config", {}) if isinstance(checkpoint, Mapping) else {}
    checkpoint_dtm = checkpoint_train_pre.get("dtm", {}) if isinstance(checkpoint_train_pre, Mapping) else {}

    testsize = int(
        _resolve(
            "testsize",
            args.testsize,
            checkpoint_train_pre.get("trainsize"),
            512,
            args.allow_preprocessing_override,
        )
    )
    dtm_scale = float(
        _resolve(
            "dtm_scale",
            args.dtm_scale,
            checkpoint_train_pre.get("dtm_scale"),
            1.0,
            args.allow_preprocessing_override,
        )
    )
    dtm_norm = _resolve(
        "dtm_norm",
        args.dtm_norm,
        checkpoint_dtm.get("mode"),
        "per_tile_zscore",
        args.allow_preprocessing_override,
    )
    dtm_divisor = float(
        _resolve(
            "dtm_divisor",
            args.dtm_divisor,
            checkpoint_dtm.get("divisor"),
            65535.0,
            args.allow_preprocessing_override,
        )
    )
    dtm_global_mean = _resolve(
        "dtm_global_mean",
        args.dtm_global_mean,
        checkpoint_dtm.get("global_mean"),
        None,
        args.allow_preprocessing_override,
    )
    dtm_global_std = _resolve(
        "dtm_global_std",
        args.dtm_global_std,
        checkpoint_dtm.get("global_std"),
        None,
        args.allow_preprocessing_override,
    )
    dtm_missing_value = float(
        _resolve(
            "dtm_missing_value",
            args.dtm_missing_value,
            checkpoint_dtm.get("missing_value"),
            0.0,
            args.allow_preprocessing_override,
        )
    )
    availability_seed = int(
        _resolve(
            "availability_seed",
            args.availability_seed,
            checkpoint_train_pre.get("availability_seed"),
            42,
            args.allow_preprocessing_override,
        )
    )
    model_cfg = _resolve(
        "model_cfg",
        args.model_cfg,
        checkpoint_run.get("model_cfg"),
        "sam2_hiera_l.yaml",
        args.allow_preprocessing_override,
    )

    dtm_availability = float(
        _resolve(
            "dtm_availability",
            args.dtm_availability,
            checkpoint_train_pre.get("dtm_availability"),
            1.0,
            args.allow_preprocessing_override,
        )
    )
    if not 0 <= dtm_availability <= 1:
        raise ValueError(f"dtm_availability must be in [0,1], got {dtm_availability}")

    checkpoint_threshold = _nested(checkpoint, "metrics", "selected_threshold", default=None)
    threshold = resolve_prediction_threshold(args.threshold, checkpoint_threshold)
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    gt_threshold = float(
        _resolve(
            "gt_threshold",
            args.gt_threshold,
            checkpoint_train_pre.get("mask_threshold"),
            0.0,
            args.allow_preprocessing_override,
        )
    )
    if not math.isfinite(gt_threshold):
        raise ValueError(f"gt_threshold must be finite, got {gt_threshold}")

    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probability_uint16"
    if args.save_probability_uint16:
        probability_dir.mkdir(exist_ok=True)

    dataset = TestDataset(
        image_root=args.test_image_path,
        mask_root=args.test_gt_path,
        dtm_root=args.test_dtm_path,
        testsize=testsize,
        dtm_scale=dtm_scale,
        dtm_availability=dtm_availability,
        availability_seed=availability_seed,
        dtm_norm=dtm_norm,
        dtm_divisor=dtm_divisor,
        dtm_global_mean=dtm_global_mean,
        dtm_global_std=dtm_global_std,
        dtm_missing_value=dtm_missing_value,
        strict_matching=not args.allow_missing_pairs,
    )
    if len(dataset) == 0:
        raise RuntimeError("No matched RGB+DTM samples found")

    from SAM2UNet import SAM2UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    model = SAM2UNet(checkpoint_path=args.hiera_path, model_cfg=model_cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    train_availability = checkpoint_train_pre.get("dtm_availability")
    print("=" * 78)
    print(f"[Checkpoint] {checkpoint_path}")
    print(f"[Device] {device}; AMP={amp_enabled}")
    print(f"[Model config] {model_cfg}; input size={testsize}")
    print(f"[DTM] norm={dtm_norm}; scale={dtm_scale}; test availability={dtm_availability:.2%}")
    if train_availability is not None:
        print(f"[DTM] checkpoint train availability={float(train_availability):.2%}")
        if float(train_availability) != dtm_availability:
            print("[DTM] NOTE: test availability differs from training; this is an explicit ablation.")
    print(f"[Threshold] prediction={threshold}; ground truth raw value > {gt_threshold}")
    if args.min_component_area > 0:
        print(f"[Post-process] remove components smaller than {args.min_component_area}px")
    if args.erode_kernel > 0:
        print(
            f"[Post-process] op={args.morph_op}; kernel={args.erode_kernel}; "
            f"iterations={args.erode_iter}"
        )
    if args.min_component_area <= 0 and args.erode_kernel <= 0:
        print("[Post-process] none")
    print(f"[Samples] {len(dataset)}")
    print("=" * 78)

    total_predicted_pixels = 0
    counts = BinaryCounts()
    per_sample: list[dict[str, object]] = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            inputs = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, _, _ = model(inputs)
            logits = F.interpolate(
                logits,
                size=tuple(sample["original_size"]),
                mode="bilinear",
                align_corners=False,
            )
            probability = torch.sigmoid(logits)[0, 0].float().cpu()
            if not torch.isfinite(probability).all():
                raise FloatingPointError(f"Non-finite prediction: {sample['name']}")

            binary = probability >= threshold
            binary_uint8 = binary.numpy().astype(np.uint8) * 255
            if args.min_component_area > 0:
                binary_uint8 = remove_small_components(binary_uint8, args.min_component_area)
                binary = torch.from_numpy(binary_uint8 > 0)
            if args.erode_kernel > 0:
                binary_uint8 = apply_morphology(
                    binary_uint8, args.erode_kernel, args.erode_iter, args.morph_op
                )
                # Keep the tensor used for metrics/pixel counts in sync with what
                # actually gets written to disk, instead of scoring the pre-morphology mask.
                binary = torch.from_numpy(binary_uint8 > 0)
            predicted_pixels = int(binary.sum().item())
            total_predicted_pixels += predicted_pixels

            save_name = f"{sample['stem']}.png"
            imageio.imwrite(output_dir / save_name, binary_uint8)
            if args.save_probability_uint16:
                probability_uint16 = np.round(probability.numpy() * 65535.0).astype(np.uint16)
                imageio.imwrite(probability_dir / save_name, probability_uint16)

            if sample["gt"] is not None:
                gt = np.asarray(sample["gt"])
                if gt.shape != tuple(sample["original_size"]):
                    raise ValueError(
                        f"GT/RGB size mismatch for {sample['name']}: GT={gt.shape}, "
                        f"RGB={sample['original_size']}"
                    )
                counts.update(binary, torch.from_numpy(gt > gt_threshold))

            record = {
                "name": sample["name"],
                "dtm_available": bool(sample["dtm_available"]),
                "probability_min": float(probability.min().item()),
                "probability_max": float(probability.max().item()),
                "probability_mean": float(probability.mean().item()),
                "predicted_foreground_pixels": predicted_pixels,
            }
            per_sample.append(record)
            print(
                f"[{index + 1}/{len(dataset)}] {save_name} "
                f"p=[{record['probability_min']:.4f},{record['probability_max']:.4f}] "
                f"mean={record['probability_mean']:.4f} foreground={predicted_pixels}"
            )

    manifest: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
        "model_cfg": model_cfg,
        "device": str(device),
        "testsize": testsize,
        "threshold": threshold,
        "dtm_scale": dtm_scale,
        "dtm_availability_test": dtm_availability,
        "dtm_availability_train": train_availability,
        "availability_seed": availability_seed,
        "dtm_normalization": {
            "mode": dtm_norm,
            "divisor": dtm_divisor,
            "global_mean": dtm_global_mean,
            "global_std": dtm_global_std,
            "missing_value": dtm_missing_value,
        },
        "ground_truth_raw_value_threshold": gt_threshold,
        "post_processing": {
            "min_component_area": args.min_component_area,
            "op": args.morph_op if args.erode_kernel > 0 else None,
            "kernel_size": args.erode_kernel,
            "iterations": args.erode_iter,
        },
        "samples": len(dataset),
        "total_predicted_foreground_pixels": total_predicted_pixels,
        "metrics": counts.metrics() if args.test_gt_path else None,
        "per_sample": per_sample,
    }
    with (output_dir / "inference_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print("=" * 78)
    print(f"[Done] masks={output_dir}; predicted foreground pixels={total_predicted_pixels}")
    if args.test_gt_path:
        print("[Fixed-threshold metrics]", counts.metrics())
    print("=" * 78)


if __name__ == "__main__":
    main(build_parser().parse_args())