import importlib

import torch

from training_utils import BinaryCounts, ThresholdMetricAccumulator, deep_supervision_loss


def test_train_module_import_does_not_parse_cli():
    module = importlib.import_module("train")
    assert callable(module.build_parser)
    assert callable(module.main)


def test_deep_supervision_loss_is_finite_and_normalized():
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 10:12, 8:20] = 1
    predictions = [torch.zeros_like(target, requires_grad=True) for _ in range(3)]
    loss = deep_supervision_loss(predictions, target, weights=(1.0, 0.5, 0.25))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(prediction.grad is not None for prediction in predictions)


def test_background_collapse_has_zero_foreground_iou_when_gt_positive():
    accumulator = ThresholdMetricAccumulator([0.5])
    logits = torch.full((1, 1, 4, 4), -20.0)
    target = torch.zeros_like(logits)
    target[..., 1, 1] = 1
    accumulator.update_logits(logits, target)
    metrics = accumulator.at(0.5)
    assert metrics["iou"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["fn"] == 1


def test_binary_counts_pooled_metric():
    counts = BinaryCounts(tp=5, fp=3, fn=2, tn=10)
    metrics = counts.metrics()
    assert metrics["iou"] == 0.5
    assert metrics["dice"] == 10 / 15


def test_training_engine_smoke_cpu():
    from torch.utils.data import DataLoader, Dataset
    from train import _make_scaler, build_optimizer, run_one_epoch

    class TinyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            image = torch.randn(4, 32, 32)
            label = torch.zeros(1, 32, 32)
            label[:, 10:12, 8:20] = 1
            return {"image": image, "label": label}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, 1)
            self.register_buffer("rgb_reference", self.conv.weight[:, :3].detach().clone())

        @property
        def dtm_conv(self):
            return self.conv

        def restore_frozen_rgb_weights(self):
            with torch.no_grad():
                self.conv.weight[:, :3].copy_(self.rgb_reference)

        def forward(self, x):
            logits = self.conv(x)
            return logits, logits, logits

    model = TinyModel()
    optimizer = build_optimizer(model, 1e-3, 0.0)
    result = run_one_epoch(
        model=model,
        loader=DataLoader(TinyDataset(), batch_size=2),
        device=torch.device("cpu"),
        train_mode=True,
        thresholds=[0.5],
        optimizer=optimizer,
        scaler=_make_scaler(False),
        amp_enabled=False,
        grad_clip=1.0,
        pos_weight=torch.tensor(1.0),
        deep_supervision_weights=(1.0, 0.5, 0.25),
    )
    assert result["loss"] > 0
    assert result["dtm_grad_norm_mean"] is not None
    assert len(result["threshold_table"]) == 1
