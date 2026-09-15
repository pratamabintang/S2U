"""Quantify checkpoint collapse and train/eval mode differences on validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FullDataset
from training_utils import BinaryCounts


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hiera_path", required=True)
    p.add_argument("--val_image_path", required=True)
    p.add_argument("--val_mask_path", required=True)
    p.add_argument("--val_dtm_path", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--max_batches", type=int, default=0, help="0 = all")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--dtm_availability", type=float, default=None)
    return p


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def nested(value: Any, *keys, default=None):
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def main(args: argparse.Namespace) -> None:
    checkpoint = torch_load(Path(args.checkpoint))
    state = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    pre = nested(checkpoint, "data_config", "train_preprocessing", default={}) or {}
    dtm_cfg = pre.get("dtm", {})
    run_cfg = checkpoint.get("run_config", {}) if isinstance(checkpoint, Mapping) else {}
    threshold = float(
        args.threshold
        if args.threshold is not None
        else nested(checkpoint, "metrics", "selected_threshold", default=0.5)
    )
    availability = float(
        args.dtm_availability
        if args.dtm_availability is not None
        else pre.get("dtm_availability", 1.0)
    )

    dataset = FullDataset(
        args.val_image_path,
        args.val_mask_path,
        args.val_dtm_path,
        trainsize=int(pre.get("trainsize", 512)),
        mode="val",
        dtm_scale=float(pre.get("dtm_scale", 1.0)),
        dtm_availability=availability,
        availability_seed=int(pre.get("availability_seed", 42)),
        dtm_norm=dtm_cfg.get("mode", "per_tile_zscore"),
        dtm_divisor=float(dtm_cfg.get("divisor", 65535.0)),
        dtm_global_mean=dtm_cfg.get("global_mean"),
        dtm_global_std=dtm_cfg.get("global_std"),
        dtm_missing_value=float(dtm_cfg.get("missing_value", 0.0)),
        mask_threshold=float(pre.get("mask_threshold", 0.0)),
        strict_matching=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    from SAM2UNet import SAM2UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet(
        args.hiera_path,
        model_cfg=run_cfg.get("model_cfg", "sam2_hiera_l.yaml"),
    ).to(device)
    model.load_state_dict(state, strict=True)

    norm_counts = {
        "BatchNorm2d": sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()),
        "GroupNorm": sum(isinstance(module, torch.nn.GroupNorm) for module in model.modules()),
    }
    counts = BinaryCounts()
    probability_means = []
    probability_minima = []
    probability_maxima = []
    mode_differences = []
    target_positive_tiles = 0
    predicted_positive_tiles = 0
    processed = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            image = batch["image"].to(device)
            target = batch["label"].to(device)

            model.eval()
            eval_logits, _, _ = model(image)
            eval_probability = torch.sigmoid(eval_logits)

            # For GroupNorm-only models, train/eval should be numerically close.
            # Do not switch a BatchNorm checkpoint to train mode because that
            # would mutate its running statistics during diagnosis.
            if norm_counts["BatchNorm2d"] == 0:
                model.train()
                train_logits, _, _ = model(image)
                mode_differences.append(float((train_logits - eval_logits).abs().max().item()))
                model.eval()

            binary = eval_probability >= threshold
            counts.update(binary, target > 0.5)
            probability_means.append(float(eval_probability.mean().item()))
            probability_minima.append(float(eval_probability.min().item()))
            probability_maxima.append(float(eval_probability.max().item()))
            target_positive_tiles += int((target.flatten(1).sum(dim=1) > 0).sum().item())
            predicted_positive_tiles += int((binary.flatten(1).sum(dim=1) > 0).sum().item())
            processed += int(image.shape[0])

    metrics = counts.metrics()
    predicted_ratio = float(metrics["predicted_foreground_ratio"])
    if predicted_ratio <= 1e-5:
        collapse = "background collapse"
    elif predicted_ratio >= 0.95:
        collapse = "foreground collapse"
    else:
        collapse = "no global all-background/all-foreground collapse detected"

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
        "device": str(device),
        "samples_processed": processed,
        "threshold": threshold,
        "dtm_availability": availability,
        "normalization_modules": norm_counts,
        "fixed_threshold_metrics": metrics,
        "target_positive_tiles": target_positive_tiles,
        "predicted_positive_tiles": predicted_positive_tiles,
        "probability": {
            "mean_of_batch_means": float(np.mean(probability_means)),
            "minimum": float(np.min(probability_minima)),
            "maximum": float(np.max(probability_maxima)),
        },
        "max_train_eval_logit_difference": (
            float(np.max(mode_differences)) if mode_differences else None
        ),
        "collapse_classification": collapse,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(parser().parse_args())
