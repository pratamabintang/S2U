"""Train SAM2-UNet on four-channel RGB+DTM inputs.

Key safeguards in this version:
- parser is not executed at import time;
- pairing is strict and case-insensitive;
- train/validation overlap is rejected;
- seeds and preprocessing are stored in checkpoints;
- validation reports IoU/Dice/precision/recall plus a threshold sweep;
- RGB patch-embedding weights are restored after every optimiser step;
- an existing log cannot be accidentally appended without --resume/--overwrite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import FullDataset
from training_utils import (
    ThresholdMetricAccumulator,
    audit_masks,
    dataset_manifest_hash,
    deep_supervision_loss,
    seed_worker,
    set_global_seed,
    split_overlap,
)


LOG_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_iou",
    "train_dice",
    "train_precision",
    "train_recall",
    "val_loss",
    "val_iou",
    "val_dice",
    "val_precision",
    "val_recall",
    "val_best_threshold",
    "val_best_iou",
    "dtm_weight_mean",
    "dtm_weight_std",
    "dtm_weight_l2",
    "dtm_grad_norm_mean",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM2-UNet RGB (8-bit) + DTM (single-band TIF), 4-channel input"
    )
    parser.add_argument("--hiera_path", required=True, help="SAM2 pretrained checkpoint")
    parser.add_argument("--model_cfg", default="sam2_hiera_l.yaml")
    parser.add_argument("--train_image_path", required=True)
    parser.add_argument("--train_mask_path", required=True)
    parser.add_argument("--train_dtm_path", required=True)
    parser.add_argument("--val_image_path", default="")
    parser.add_argument("--val_mask_path", default="")
    parser.add_argument("--val_dtm_path", default="")
    parser.add_argument("--save_path", required=True)

    # --epoch remains an alias so old shell commands do not break.  The new
    # semantics are TOTAL target epochs, including epochs already in --resume.
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_weights_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--trainsize", type=int, default=512)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--allow_missing_pairs", action="store_true")

    parser.add_argument("--dtm_scale", type=float, default=1.0)
    parser.add_argument("--dtm_availability", type=float, default=1.0)
    parser.add_argument(
        "--val_dtm_availability",
        type=float,
        default=None,
        help="Defaults to --dtm_availability. Set separately for controlled evaluation.",
    )
    parser.add_argument("--availability_seed", type=int, default=42)
    parser.add_argument(
        "--dtm_norm",
        choices=("per_tile_zscore", "uint16_01", "global_zscore", "none"),
        default="per_tile_zscore",
    )
    parser.add_argument("--dtm_divisor", type=float, default=65535.0)
    parser.add_argument("--dtm_global_mean", type=float, default=None)
    parser.add_argument("--dtm_global_std", type=float, default=None)
    parser.add_argument("--dtm_missing_value", type=float, default=0.0)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold_sweep",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--auto_pos_weight", action="store_true")
    parser.add_argument("--max_auto_pos_weight", type=float, default=30.0)
    parser.add_argument(
        "--deep_supervision_weights", type=float, nargs=3, default=[1.0, 0.5, 0.25]
    )

    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatic mixed precision on CUDA (use --no-amp to disable).",
    )
    parser.add_argument("--skip_label_audit", action="store_true")
    parser.add_argument("--debug_probability_batches", type=int, default=0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("hiera_path", "train_image_path", "train_mask_path", "train_dtm_path"):
        path = Path(getattr(args, name))
        expected = path.is_file() if name == "hiera_path" else path.is_dir()
        if not expected:
            kind = "file" if name == "hiera_path" else "directory"
            raise FileNotFoundError(f"--{name} {kind} not found: {path}")

    supplied_validation = [bool(args.val_image_path), bool(args.val_mask_path), bool(args.val_dtm_path)]
    if any(supplied_validation) and not all(supplied_validation):
        raise ValueError("Provide all three validation paths or none of them")
    if all(supplied_validation):
        for name in ("val_image_path", "val_mask_path", "val_dtm_path"):
            if not Path(getattr(args, name)).is_dir():
                raise FileNotFoundError(f"--{name} directory not found: {getattr(args, name)}")

    if args.epochs <= 0 or args.batch_size <= 0 or args.trainsize <= 0:
        raise ValueError("epochs, batch_size and trainsize must be > 0")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be > 0 and weight_decay must be >= 0")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if not 0 <= args.dtm_availability <= 1:
        raise ValueError("dtm_availability must be in [0,1]")
    if args.val_dtm_availability is not None and not 0 <= args.val_dtm_availability <= 1:
        raise ValueError("val_dtm_availability must be in [0,1]")
    if args.pos_weight <= 0 or args.max_auto_pos_weight <= 0:
        raise ValueError("pos_weight values must be > 0")
    if args.patience < 0:
        raise ValueError("patience must be >= 0")


def make_dataset(
    *,
    image_root: str,
    mask_root: str,
    dtm_root: str,
    mode: str,
    args: argparse.Namespace,
    availability: float,
) -> FullDataset:
    return FullDataset(
        image_root=image_root,
        mask_root=mask_root,
        dtm_root=dtm_root,
        trainsize=args.trainsize,
        mode=mode,
        dtm_scale=args.dtm_scale,
        dtm_availability=availability,
        availability_seed=args.availability_seed,
        dtm_norm=args.dtm_norm,
        dtm_divisor=args.dtm_divisor,
        dtm_global_mean=args.dtm_global_mean,
        dtm_global_std=args.dtm_global_std,
        dtm_missing_value=args.dtm_missing_value,
        mask_threshold=args.mask_threshold,
        strict_matching=not args.allow_missing_pairs,
        return_metadata=True,
    )


def make_loader(
    dataset: FullDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_mode: bool,
    thresholds: Sequence[float],
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    amp_enabled: bool,
    grad_clip: float,
    pos_weight: Optional[torch.Tensor],
    deep_supervision_weights: Sequence[float],
    debug_probability_batches: int = 0,
) -> Dict[str, object]:
    model.train(train_mode)
    accumulator = ThresholdMetricAccumulator(thresholds)
    loss_sum = 0.0
    sample_count = 0
    grad_norms: List[float] = []
    debug_printed = 0

    for batch_index, batch in enumerate(loader):
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError(f"Expected [B,4,H,W], got {tuple(inputs.shape)}")
        if targets.ndim != 4 or targets.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(targets.shape)}")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise FloatingPointError(f"Non-finite batch at index {batch_index}")

        if train_mode:
            if optimizer is None:
                raise ValueError("optimizer is required in train mode")
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predictions = model(inputs)
                loss = deep_supervision_loss(
                    predictions,
                    targets,
                    weights=deep_supervision_weights,
                    pos_weight=pos_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {batch_index}: {loss.item()}")

            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                dtm_gradient = model.dtm_conv.weight.grad
                if dtm_gradient is not None:
                    grad_norms.append(float(dtm_gradient[:, 3:4].norm().item()))
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        max_norm=grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                model.restore_frozen_rgb_weights()

        batch_size = int(inputs.shape[0])
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        accumulator.update_logits(predictions[0], targets)

        if debug_printed < debug_probability_batches and (targets > 0.5).any():
            probabilities = torch.sigmoid(predictions[0].detach())
            print(
                f"[Probability debug batch={batch_index}] min={probabilities.min().item():.6f} "
                f"max={probabilities.max().item():.6f} mean={probabilities.mean().item():.6f} "
                f"pred>0.5={(probabilities >= 0.5).float().mean().item():.6f} "
                f"target_fg={(targets > 0.5).float().mean().item():.6f}"
            )
            debug_printed += 1

    return {
        "loss": loss_sum / max(sample_count, 1),
        "threshold_table": accumulator.table(),
        "best_threshold": accumulator.best("iou"),
        "dtm_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
    }


def metric_at(result: Mapping[str, object], threshold: float) -> Dict[str, float | int]:
    table = result["threshold_table"]
    return min(table, key=lambda row: abs(float(row["threshold"]) - threshold))


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    run_config: Mapping[str, object],
    data_config: Mapping[str, object],
    metrics: Mapping[str, object],
    best_val_iou: float,
    best_val_loss: float,
) -> None:
    checkpoint = {
        "schema_version": 2,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "run_config": dict(run_config),
        "data_config": dict(data_config),
        "metrics": dict(metrics),
        "best_val_iou": float(best_val_iou),
        "best_val_loss": float(best_val_loss),
    }
    torch.save(checkpoint, path)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def append_log(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in LOG_FIELDS})


def save_plots(csv_path: Path, output_directory: Path) -> None:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]

    plt.figure()
    plt.plot(epochs, [float(row["train_loss"]) for row in rows], label="train_loss")
    if all(row["val_loss"] != "" for row in rows):
        plt.plot(epochs, [float(row["val_loss"]) for row in rows], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "loss_vs_epoch.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [float(row["train_iou"]) for row in rows], label="train_online_iou")
    if all(row["val_iou"] != "" for row in rows):
        plt.plot(epochs, [float(row["val_iou"]) for row in rows], label="val_iou")
    plt.xlabel("epoch")
    plt.ylabel("pooled foreground IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "iou_vs_epoch.png", dpi=200)
    plt.close()


def prepare_output_directory(save_path: Path, resume: Optional[str], overwrite: bool) -> Path:
    save_path.mkdir(parents=True, exist_ok=True)
    expected_header = ",".join(LOG_FIELDS)
    log_path = save_path / "log.csv"

    if resume and log_path.exists():
        existing_header = log_path.open("r", encoding="utf-8").readline().strip()
        if existing_header != expected_header:
            # Do not append v2 rows underneath the old notebook's different CSV
            # schema. Preserve it and start a clearly named continuation log.
            log_path = save_path / "log_v2.csv"
            print("[WARNING] Existing log.csv uses the legacy schema; continuing in log_v2.csv")
            if log_path.exists():
                v2_header = log_path.open("r", encoding="utf-8").readline().strip()
                if v2_header != expected_header:
                    raise RuntimeError(f"Unexpected header in {log_path}")

    if log_path.exists() and not resume:
        if not overwrite:
            raise FileExistsError(
                f"{log_path} already exists. Use --resume to continue or --overwrite for a new run."
            )
        for path in save_path.glob("*"):
            if path.is_file() and (
                path.suffix in {".csv", ".json", ".png", ".pth"}
                or path.name.startswith("best_")
            ):
                path.unlink()
    return log_path


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    set_global_seed(args.seed, deterministic=args.deterministic)
    save_dir = Path(args.save_path)
    log_path = prepare_output_directory(save_dir, args.resume, args.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"[Device] {device}; AMP={amp_enabled}; deterministic={args.deterministic}; seed={args.seed}")

    val_availability = (
        args.dtm_availability if args.val_dtm_availability is None else args.val_dtm_availability
    )
    train_dataset = make_dataset(
        image_root=args.train_image_path,
        mask_root=args.train_mask_path,
        dtm_root=args.train_dtm_path,
        mode="train",
        args=args,
        availability=args.dtm_availability,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset has zero matched samples")

    val_dataset = None
    if args.val_image_path:
        val_dataset = make_dataset(
            image_root=args.val_image_path,
            mask_root=args.val_mask_path,
            dtm_root=args.val_dtm_path,
            mode="val",
            args=args,
            availability=val_availability,
        )
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset has zero matched samples")
        overlap = split_overlap(train_dataset.stems, val_dataset.stems)
        if overlap:
            raise RuntimeError(
                f"Data leakage: {len(overlap)} stems occur in train and validation, e.g. {overlap[:10]}"
            )

    audits: Dict[str, object] = {
        "train_pairing": train_dataset.pairing_report.to_dict(),
        "train_manifest_sha256": dataset_manifest_hash(train_dataset.stems),
    }
    train_audit = None
    if not args.skip_label_audit:
        train_audit = audit_masks(train_dataset.gts, threshold=args.mask_threshold)
        if int(train_audit["foreground_pixels"]) == 0:
            raise RuntimeError("Training masks contain zero foreground pixels at the configured threshold")
        audits["train_masks"] = train_audit
        print("[Train mask audit]", json.dumps(train_audit, ensure_ascii=False))
    if val_dataset is not None:
        audits["val_pairing"] = val_dataset.pairing_report.to_dict()
        audits["val_manifest_sha256"] = dataset_manifest_hash(val_dataset.stems)
        if not args.skip_label_audit:
            val_audit = audit_masks(val_dataset.gts, threshold=args.mask_threshold)
            audits["val_masks"] = val_audit
            print("[Validation mask audit]", json.dumps(val_audit, ensure_ascii=False))
            train_ratio = float(train_audit["foreground_ratio"])
            val_ratio = float(val_audit["foreground_ratio"])
            ratio_shift = max(train_ratio, val_ratio) / max(min(train_ratio, val_ratio), 1e-12)
            audits["foreground_ratio_shift_factor"] = ratio_shift
            if ratio_shift >= 2:
                print(f"[WARNING] Train/validation foreground-pixel ratio differs by {ratio_shift:.2f}x")
    write_json(save_dir / "dataset_audit.json", audits)

    pos_weight_value = args.pos_weight
    if args.auto_pos_weight:
        if train_audit is None:
            raise ValueError("--auto_pos_weight requires label audit; remove --skip_label_audit")
        raw = float(train_audit["suggested_raw_pos_weight"])
        pos_weight_value = min(raw, args.max_auto_pos_weight)
        print(f"[pos_weight] raw background/foreground={raw:.4f}; applied={pos_weight_value:.4f}")
    pos_weight = torch.tensor(pos_weight_value, device=device, dtype=torch.float32)

    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, True, device, args.seed
    )
    val_loader = (
        make_loader(val_dataset, args.batch_size, args.num_workers, False, device, args.seed + 1)
        if val_dataset is not None
        else None
    )

    from SAM2UNet import SAM2UNet, report_dtm_weight_stats

    model = SAM2UNet(
        checkpoint_path=args.hiera_path,
        model_cfg=args.model_cfg,
    ).to(device)
    print("[Parameters]", model.trainable_parameter_report())

    sample = train_dataset[0]["image"].unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        smoke_outputs = model(sample)
        output_shapes = [tuple(output.shape) for output in smoke_outputs]
    print(f"[Smoke test] input={tuple(sample.shape)} outputs={output_shapes}")
    if any(output.shape[-2:] != sample.shape[-2:] for output in smoke_outputs):
        raise RuntimeError("Model output size differs from input size")
    # Accessing a training sample can consume Python RNG when num_workers=0.
    # Restore the declared seed so the smoke test does not change augmentation
    # order in the actual optimisation run.
    set_global_seed(args.seed, deterministic=args.deterministic)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-7)
    scaler = _make_scaler(amp_enabled)

    start_epoch = 1
    best_val_iou = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        if isinstance(checkpoint, dict):
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_iou = float(checkpoint.get("best_val_iou", -1.0))
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            if not args.resume_weights_only:
                try:
                    if "optim" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optim"])
                    if "sched" in checkpoint:
                        scheduler.load_state_dict(checkpoint["sched"])
                    if "scaler" in checkpoint:
                        scaler.load_state_dict(checkpoint["scaler"])
                except (ValueError, KeyError) as exc:
                    print(f"[WARNING] Optimiser/scheduler state incompatible; continuing with fresh state: {exc}")
        print(f"[Resume] loaded {resume_path}; next epoch={start_epoch}")

    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch - 1}, beyond target --epochs {args.epochs}"
        )

    thresholds = sorted(set(float(value) for value in args.threshold_sweep + [args.threshold]))
    run_config = vars(args).copy()
    run_config["resolved_device"] = str(device)
    run_config["amp_enabled"] = amp_enabled
    run_config["resolved_pos_weight"] = pos_weight_value
    run_config["val_dtm_availability_resolved"] = val_availability
    data_config = {
        "train_preprocessing": train_dataset.preprocessing_config(),
        "val_preprocessing": val_dataset.preprocessing_config() if val_dataset else None,
        "train_manifest_sha256": audits["train_manifest_sha256"],
        "val_manifest_sha256": audits.get("val_manifest_sha256"),
    }
    write_json(save_dir / "run_config.json", {"run_config": run_config, "data_config": data_config})

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_result = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            train_mode=True,
            thresholds=[args.threshold],
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip=args.grad_clip,
            pos_weight=pos_weight,
            deep_supervision_weights=args.deep_supervision_weights,
            debug_probability_batches=0,
        )
        train_metrics = metric_at(train_result, args.threshold)

        val_result = None
        val_metrics: Dict[str, object] = {}
        improved = False
        loss_improved = False
        if val_loader is not None:
            val_result = run_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                train_mode=False,
                thresholds=thresholds,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip=args.grad_clip,
                pos_weight=pos_weight,
                deep_supervision_weights=args.deep_supervision_weights,
                debug_probability_batches=args.debug_probability_batches,
            )
            val_metrics = metric_at(val_result, args.threshold)
            best_threshold_metrics = val_result["best_threshold"]
            current_iou = float(val_metrics["iou"])
            current_loss = float(val_result["loss"])
            if current_iou > best_val_iou + args.min_delta:
                best_val_iou = current_iou
                epochs_without_improvement = 0
                improved = True
            else:
                epochs_without_improvement += 1
            if current_loss < best_val_loss:
                loss_improved = True
                best_val_loss = current_loss
        else:
            best_threshold_metrics = {"threshold": args.threshold, "iou": train_metrics["iou"]}

        scheduler.step()
        dtm_stats = report_dtm_weight_stats(
            model.dtm_conv, grad_norm=train_result["dtm_grad_norm_mean"]
        )

        row = {
            "epoch": epoch,
            # LR actually used for this epoch; scheduler.step() above prepares
            # the optimiser and checkpoint state for the next epoch.
            "lr": current_lr,
            "train_loss": train_result["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_result["loss"] if val_result else "",
            "val_iou": val_metrics.get("iou", ""),
            "val_dice": val_metrics.get("dice", ""),
            "val_precision": val_metrics.get("precision", ""),
            "val_recall": val_metrics.get("recall", ""),
            "val_best_threshold": best_threshold_metrics["threshold"],
            "val_best_iou": best_threshold_metrics["iou"],
            "dtm_weight_mean": dtm_stats["mean"],
            "dtm_weight_std": dtm_stats["std"],
            "dtm_weight_l2": dtm_stats["l2"],
            "dtm_grad_norm_mean": train_result["dtm_grad_norm_mean"],
        }
        append_log(log_path, row)
        save_plots(log_path, save_dir)

        print(
            f"epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.6f} "
            f"train_iou={row['train_iou']:.6f} "
            + (
                f"val_loss={row['val_loss']:.6f} val_iou@{args.threshold:.2f}={row['val_iou']:.6f} "
                f"val_best_iou={row['val_best_iou']:.6f}@{row['val_best_threshold']:.2f}"
                if val_result
                else ""
            )
        )

        checkpoint_metrics = {
            "train": train_metrics,
            "val": val_metrics or None,
            "val_threshold_sweep": val_result["threshold_table"] if val_result else None,
            "selected_threshold": best_threshold_metrics["threshold"],
        }
        common_checkpoint_args = dict(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            run_config=run_config,
            data_config=data_config,
            metrics=checkpoint_metrics,
            best_val_iou=best_val_iou,
            best_val_loss=best_val_loss,
        )
        if val_result and improved:
            save_checkpoint(save_dir / "best_by_val_iou.pth", **common_checkpoint_args)
        if val_result and loss_improved:
            save_checkpoint(save_dir / "best_by_val_loss.pth", **common_checkpoint_args)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(save_dir / f"SAM2-UNet-{epoch}.pth", **common_checkpoint_args)

        if val_loader is not None and args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"[Early stop] no val IoU improvement for {args.patience} epochs; "
                f"best val IoU={best_val_iou:.6f}"
            )
            break

    print(f"[Done] log={log_path}; best_val_iou={best_val_iou:.6f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
