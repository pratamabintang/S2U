"""Reusable loss, metric and reproducibility helpers."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic and torch.cuda.is_available()
    if deterministic:
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def structure_loss(pred: torch.Tensor, mask: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Boundary-weighted BCE + soft IoU loss from SAM2-UNet."""

    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weighted_bce = F.binary_cross_entropy_with_logits(
        pred, mask, reduction="none", pos_weight=pos_weight
    )
    weighted_bce = (weight * weighted_bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1e-7)

    probability = torch.sigmoid(pred)
    intersection = ((probability * mask) * weight).sum(dim=(2, 3))
    union = ((probability + mask) * weight).sum(dim=(2, 3))
    weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)
    return (weighted_bce + weighted_iou).mean()


def deep_supervision_loss(
    predictions: Sequence[torch.Tensor],
    target: torch.Tensor,
    weights: Sequence[float] = (1.0, 0.5, 0.25),
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(predictions) != len(weights):
        raise ValueError(f"Got {len(predictions)} predictions but {len(weights)} loss weights")
    weight_sum = float(sum(weights))
    if weight_sum <= 0 or any(weight < 0 for weight in weights):
        raise ValueError(f"Deep-supervision weights must be non-negative with positive sum: {weights}")
    losses = [
        float(weight) * structure_loss(prediction, target, pos_weight=pos_weight)
        for prediction, weight in zip(predictions, weights)
    ]
    return sum(losses) / weight_sum


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.bool()
        target = target.bool()
        self.tp += int(torch.logical_and(prediction, target).sum().item())
        self.fp += int(torch.logical_and(prediction, ~target).sum().item())
        self.fn += int(torch.logical_and(~prediction, target).sum().item())
        self.tn += int(torch.logical_and(~prediction, ~target).sum().item())

    def metrics(self) -> Dict[str, float | int]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou_denominator = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn
        precision_denominator = tp + fp
        recall_denominator = tp + fn
        specificity_denominator = tn + fp
        total = tp + fp + fn + tn
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "iou": tp / iou_denominator if iou_denominator else 1.0,
            "dice": 2 * tp / dice_denominator if dice_denominator else 1.0,
            "precision": tp / precision_denominator if precision_denominator else 0.0,
            "recall": tp / recall_denominator if recall_denominator else 0.0,
            "specificity": tn / specificity_denominator if specificity_denominator else 0.0,
            "accuracy": (tp + tn) / total if total else 0.0,
            "predicted_foreground_ratio": (tp + fp) / total if total else 0.0,
            "target_foreground_ratio": (tp + fn) / total if total else 0.0,
        }


class ThresholdMetricAccumulator:
    def __init__(self, thresholds: Sequence[float]):
        if not thresholds:
            raise ValueError("At least one threshold is required")
        cleaned = sorted({round(float(value), 6) for value in thresholds})
        if cleaned[0] < 0 or cleaned[-1] > 1:
            raise ValueError(f"Thresholds must be in [0,1], got {cleaned}")
        self.thresholds = cleaned
        self.counts = {threshold: BinaryCounts() for threshold in cleaned}

    @torch.no_grad()
    def update_logits(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probabilities = torch.sigmoid(logits.detach())
        target_bool = target.detach() > 0.5
        for threshold in self.thresholds:
            self.counts[threshold].update(probabilities >= threshold, target_bool)

    def table(self) -> List[Dict[str, float | int]]:
        rows = []
        for threshold in self.thresholds:
            row = {"threshold": threshold}
            row.update(self.counts[threshold].metrics())
            rows.append(row)
        return rows

    def at(self, threshold: float) -> Dict[str, float | int]:
        key = min(self.thresholds, key=lambda value: abs(value - float(threshold)))
        row = {"threshold": key}
        row.update(self.counts[key].metrics())
        return row

    def best(self, metric: str = "iou") -> Dict[str, float | int]:
        rows = self.table()
        if metric not in rows[0]:
            raise KeyError(f"Unknown metric {metric!r}")
        # Prefer a threshold closer to 0.5 when scores tie.
        return max(rows, key=lambda row: (float(row[metric]), -abs(float(row["threshold"]) - 0.5)))


def audit_masks(mask_paths: Sequence[str], threshold: float = 0.0) -> Dict[str, float | int]:
    positive_tiles = 0
    foreground_pixels = 0
    total_pixels = 0
    unique_values: set[int] = set()

    for path in mask_paths:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))
        unique_values.update(int(value) for value in np.unique(mask)[:256])
        foreground = mask > threshold
        positive_tiles += int(foreground.any())
        foreground_pixels += int(foreground.sum())
        total_pixels += int(foreground.size)

    foreground_ratio = foreground_pixels / total_pixels if total_pixels else 0.0
    background_pixels = total_pixels - foreground_pixels
    # None is deliberate: strict JSON rejects Infinity, and a dataset with no
    # foreground should be treated as an integrity failure rather than as an
    # infinitely weighted optimisation target.
    suggested_pos_weight = background_pixels / foreground_pixels if foreground_pixels else None
    return {
        "files": len(mask_paths),
        "positive_tiles": positive_tiles,
        "negative_tiles": len(mask_paths) - positive_tiles,
        "foreground_pixels": foreground_pixels,
        "background_pixels": background_pixels,
        "total_pixels": total_pixels,
        "foreground_ratio": foreground_ratio,
        "suggested_raw_pos_weight": suggested_pos_weight,
        "unique_values_count": len(unique_values),
        "unique_values_preview": sorted(unique_values)[:32],
    }


def dataset_manifest_hash(stems: Sequence[str]) -> str:
    canonical = "\n".join(sorted(stem.casefold() for stem in stems))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_overlap(train_stems: Sequence[str], val_stems: Sequence[str]) -> List[str]:
    train = {stem.casefold(): stem for stem in train_stems}
    val = {stem.casefold(): stem for stem in val_stems}
    return sorted(train[key] for key in set(train) & set(val))
