"""Dataset pipeline for RGB + DTM semantic segmentation.

The implementation is deliberately strict about pairing and explicit about
pre-processing.  The old code silently dropped samples when a mask/DTM name
was not an exact case-sensitive match; that can change the class distribution
without producing an error.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DTM_EXTENSIONS = (".tif", ".tiff")


class DatasetIntegrityError(RuntimeError):
    """Raised when files cannot be paired safely."""


@dataclass(frozen=True)
class DTMNormalizationConfig:
    """DTM preprocessing parameters stored in each training checkpoint.

    Modes
    -----
    per_tile_zscore:
        Standardise each finite tile independently.  This emphasises local
        relief but removes absolute elevation.  ``divisor`` is mathematically
        cancelled by z-scoring and is retained only for an explicit record.
    uint16_01:
        Divide by ``divisor`` (normally 65535) and keep absolute scaled values.
    global_zscore:
        Divide by ``divisor`` and apply supplied train-set mean/std.
    none:
        Keep raw finite values.  Use only when the source is already scaled.
    """

    mode: str = "per_tile_zscore"
    divisor: float = 65535.0
    global_mean: Optional[float] = None
    global_std: Optional[float] = None
    eps: float = 1e-6
    missing_value: float = 0.0

    def validate(self) -> None:
        valid_modes = {"per_tile_zscore", "uint16_01", "global_zscore", "none"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown DTM normalization mode {self.mode!r}; choose {sorted(valid_modes)}")
        if not math.isfinite(self.divisor) or self.divisor <= 0:
            raise ValueError(f"DTM divisor must be finite and > 0, got {self.divisor}")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"DTM eps must be finite and > 0, got {self.eps}")
        if not math.isfinite(self.missing_value):
            raise ValueError("DTM missing_value must be finite")
        if self.mode == "global_zscore":
            if self.global_mean is None or self.global_std is None:
                raise ValueError("global_zscore requires global_mean and global_std")
            if not math.isfinite(self.global_mean):
                raise ValueError("global_mean must be finite")
            if not math.isfinite(self.global_std) or self.global_std <= self.eps:
                raise ValueError("global_std must be finite and greater than eps")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SampleRecord:
    stem: str
    image_path: str
    mask_path: Optional[str]
    dtm_path: str


@dataclass(frozen=True)
class PairingReport:
    image_count: int
    mask_count: Optional[int]
    dtm_count: int
    matched_count: int
    missing_masks: Tuple[str, ...]
    missing_dtms: Tuple[str, ...]
    orphan_masks: Tuple[str, ...]
    orphan_dtms: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _validate_directory(path: str | os.PathLike[str], label: str) -> Path:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {root}")
    return root


def _index_directory(
    root: str | os.PathLike[str],
    extensions: Sequence[str],
    label: str,
) -> Dict[str, Path]:
    """Return a case-insensitive stem index and fail on ambiguous duplicates."""

    folder = _validate_directory(root, label)
    allowed = {ext.casefold() for ext in extensions}
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = {}

    for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in allowed:
            continue
        key = path.stem.casefold()
        if key in index:
            duplicates.setdefault(key, [index[key].name]).append(path.name)
        else:
            index[key] = path

    if duplicates:
        detail = "; ".join(f"{stem}: {names}" for stem, names in sorted(duplicates.items()))
        raise DatasetIntegrityError(
            f"Ambiguous {label} filenames after case-insensitive stem matching: {detail}"
        )
    return index


def build_sample_records(
    image_root: str | os.PathLike[str],
    dtm_root: str | os.PathLike[str],
    mask_root: Optional[str | os.PathLike[str]] = None,
    *,
    strict: bool = True,
) -> Tuple[List[SampleRecord], PairingReport]:
    """Pair RGB, mask and DTM files by case-insensitive stem.

    ``strict=True`` prevents silent sample deletion.  This is intentionally the
    default because silently losing positive masks can make validation IoU look
    catastrophically low while all code continues running.
    """

    images = _index_directory(image_root, IMAGE_EXTENSIONS, "RGB")
    dtms = _index_directory(dtm_root, DTM_EXTENSIONS, "DTM")
    masks = _index_directory(mask_root, MASK_EXTENSIONS, "mask") if mask_root else None

    image_keys = set(images)
    dtm_keys = set(dtms)
    mask_keys = set(masks) if masks is not None else image_keys

    missing_dtms = sorted(image_keys - dtm_keys)
    missing_masks = sorted(image_keys - mask_keys) if masks is not None else []
    common = sorted(image_keys & dtm_keys & mask_keys)

    orphan_dtms = sorted(dtm_keys - image_keys)
    orphan_masks = sorted(mask_keys - image_keys) if masks is not None else []

    report = PairingReport(
        image_count=len(images),
        mask_count=len(masks) if masks is not None else None,
        dtm_count=len(dtms),
        matched_count=len(common),
        missing_masks=tuple(missing_masks),
        missing_dtms=tuple(missing_dtms),
        orphan_masks=tuple(orphan_masks),
        orphan_dtms=tuple(orphan_dtms),
    )

    if strict and (missing_masks or missing_dtms):
        parts = []
        if missing_masks:
            parts.append(f"missing mask for {len(missing_masks)} RGB stems: {missing_masks[:10]}")
        if missing_dtms:
            parts.append(f"missing DTM for {len(missing_dtms)} RGB stems: {missing_dtms[:10]}")
        raise DatasetIntegrityError("; ".join(parts))

    records = [
        SampleRecord(
            stem=images[key].stem,
            image_path=str(images[key]),
            mask_path=str(masks[key]) if masks is not None and key in masks else None,
            dtm_path=str(dtms[key]),
        )
        for key in common
    ]
    return records, report


def deterministic_availability_mask(
    stems: Sequence[str], availability: float, seed: int
) -> List[bool]:
    """Select exactly round(N * availability) tiles using stable hash scores.

    The selected sets are nested across 0/25/50/75/100% runs and independent
    of directory listing order, making the ablation controlled and repeatable.
    """

    if not 0.0 <= availability <= 1.0:
        raise ValueError(f"dtm_availability must be in [0, 1], got {availability}")

    n_selected = int(round(len(stems) * availability))
    scored: List[Tuple[int, int]] = []
    for index, stem in enumerate(stems):
        digest = hashlib.sha256(f"{seed}:{stem.casefold()}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], byteorder="big", signed=False)
        scored.append((score, index))
    selected_indices = {index for _, index in sorted(scored)[:n_selected]}
    return [index in selected_indices for index in range(len(stems))]


def read_dtm_masked(path: str | os.PathLike[str]) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read band 1 with rasterio's mask handling, including NaN nodata."""

    with rasterio.open(path) as source:
        band = source.read(1, masked=True)
        # Convert the masked array *before* filling with NaN. Calling
        # ``filled(np.nan)`` directly on an integer GeoTIFF (common for
        # 16-bit DTM rasters) raises ``TypeError: Cannot convert fill_value
        # nan to dtype uint16`` even when no pixels are masked.
        array = np.asarray(band.astype(np.float32).filled(np.nan), dtype=np.float32)
        nodata_raw = source.nodata
        nodata_is_nan = bool(
            nodata_raw is not None and not math.isfinite(float(nodata_raw))
        )
        # Keep metadata strict-JSON-safe. A literal NaN would make audit JSON
        # fail under allow_nan=False even though raster masking succeeded.
        nodata = None if nodata_raw is None or nodata_is_nan else float(nodata_raw)
        metadata = {
            "dtype": str(source.dtypes[0]),
            "nodata": nodata,
            "nodata_is_nan": nodata_is_nan,
            "shape": (source.height, source.width),
            "count": source.count,
            "crs": str(source.crs) if source.crs is not None else None,
            "transform": tuple(source.transform),
        }
    return array, metadata


def normalize_dtm(array: np.ndarray, config: DTMNormalizationConfig) -> np.ndarray:
    config.validate()
    dtm = np.asarray(array, dtype=np.float32).copy()
    finite = np.isfinite(dtm)
    if not finite.any():
        return dtm

    if config.mode == "none":
        return dtm

    scaled = dtm / float(config.divisor)
    if config.mode == "uint16_01":
        return scaled

    if config.mode == "per_tile_zscore":
        mean = float(scaled[finite].mean())
        std = float(scaled[finite].std())
        scaled[finite] = (scaled[finite] - mean) / max(std, config.eps)
        return scaled

    # global_zscore
    scaled[finite] = (scaled[finite] - float(config.global_mean)) / float(config.global_std)
    return scaled


def resize_dtm_nan_safe(array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinearly resize finite values without spreading NaNs into neighbours."""

    dtm = np.asarray(array, dtype=np.float32)
    if tuple(dtm.shape) == tuple(size):
        return dtm.copy()

    finite = np.isfinite(dtm)
    values = np.where(finite, dtm, 0.0).astype(np.float32)
    weights = finite.astype(np.float32)

    values_t = torch.from_numpy(values)[None, None]
    weights_t = torch.from_numpy(weights)[None, None]

    resized_values = F.interpolate(values_t, size=size, mode="bilinear", align_corners=False)
    resized_weights = F.interpolate(weights_t, size=size, mode="bilinear", align_corners=False)

    valid = resized_weights > 1e-6
    result = torch.full_like(resized_values, float("nan"))
    result[valid] = resized_values[valid] / resized_weights[valid]
    return result[0, 0].numpy()


def load_rgb(path: str | os.PathLike[str]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_mask(path: str | os.PathLike[str]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").copy()


class FullDataset(Dataset):
    """Training/validation dataset returning ``image=[4,H,W]`` and mask."""

    def __init__(
        self,
        image_root: str,
        mask_root: str,
        dtm_root: str,
        trainsize: int,
        mode: str,
        dtm_scale: float = 1.0,
        dtm_availability: float = 1.0,
        availability_seed: int = 42,
        dtm_norm: str = "per_tile_zscore",
        dtm_divisor: float = 65535.0,
        dtm_global_mean: Optional[float] = None,
        dtm_global_std: Optional[float] = None,
        dtm_missing_value: float = 0.0,
        mask_threshold: float = 0.0,
        strict_matching: bool = True,
        return_metadata: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be train/val/test, got {mode!r}")
        if int(trainsize) <= 0:
            raise ValueError("trainsize must be > 0")
        if not math.isfinite(float(dtm_scale)):
            raise ValueError("dtm_scale must be finite")
        if not math.isfinite(float(mask_threshold)):
            raise ValueError("mask_threshold must be finite")

        self.records, self.pairing_report = build_sample_records(
            image_root=image_root,
            mask_root=mask_root,
            dtm_root=dtm_root,
            strict=strict_matching,
        )
        self.mode = mode
        self.trainsize = int(trainsize)
        self.size = (self.trainsize, self.trainsize)
        self.dtm_scale = float(dtm_scale)
        self.dtm_availability = float(dtm_availability)
        self.availability_seed = int(availability_seed)
        self.mask_threshold = float(mask_threshold)
        self.return_metadata = bool(return_metadata)
        self.dtm_config = DTMNormalizationConfig(
            mode=dtm_norm,
            divisor=float(dtm_divisor),
            global_mean=dtm_global_mean,
            global_std=dtm_global_std,
            missing_value=float(dtm_missing_value),
        )
        self.dtm_config.validate()

        self.stems = [record.stem for record in self.records]
        self.images = [record.image_path for record in self.records]
        self.gts = [str(record.mask_path) for record in self.records]
        self.dtms = [record.dtm_path for record in self.records]
        self.dtm_available_mask = deterministic_availability_mask(
            self.stems, self.dtm_availability, self.availability_seed
        )

        n_available = sum(self.dtm_available_mask)
        print(f"[Dataset:{mode}] matched={len(self.records)} / RGB={self.pairing_report.image_count}")
        print(
            f"[Dataset:{mode}] DTM availability={n_available}/{len(self.records)} "
            f"({n_available / max(len(self.records), 1):.2%}), scale={self.dtm_scale}"
        )
        print(f"[Dataset:{mode}] RGB norm=ImageNet; DTM norm={self.dtm_config.to_dict()}")

    def __len__(self) -> int:
        return len(self.records)

    def _prepare_dtm(self, record: SampleRecord, available: bool) -> np.ndarray:
        if not available:
            return np.full(self.size, self.dtm_config.missing_value, dtype=np.float32)

        raw, _ = read_dtm_masked(record.dtm_path)
        normalised = normalize_dtm(raw, self.dtm_config)
        resized = resize_dtm_nan_safe(normalised, self.size)
        return np.nan_to_num(
            resized,
            nan=self.dtm_config.missing_value,
            posinf=self.dtm_config.missing_value,
            neginf=self.dtm_config.missing_value,
        ).astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        image = load_rgb(record.image_path)
        if record.mask_path is None:
            raise DatasetIntegrityError(f"Training record has no mask: {record.stem}")
        label = load_mask(record.mask_path)

        if image.size != label.size:
            raise DatasetIntegrityError(
                f"RGB/mask size mismatch for {record.stem}: RGB={image.size}, mask={label.size}"
            )

        image = TF.resize(
            image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        label = TF.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        dtm = self._prepare_dtm(record, self.dtm_available_mask[index])

        if self.mode == "train":
            if random.random() < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                dtm = np.fliplr(dtm).copy()
            if random.random() < 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                dtm = np.flipud(dtm).copy()

        rgb_tensor = TF.to_tensor(image)
        rgb_tensor = TF.normalize(rgb_tensor, IMAGENET_MEAN, IMAGENET_STD)

        label_array = np.asarray(label, dtype=np.float32)
        label_tensor = torch.from_numpy((label_array > self.mask_threshold).astype(np.float32))[None]

        dtm_tensor = torch.from_numpy(dtm)[None] * self.dtm_scale
        input_tensor = torch.cat((rgb_tensor, dtm_tensor), dim=0)

        if input_tensor.shape[0] != 4 or label_tensor.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected tensor shape for {record.stem}: image={tuple(input_tensor.shape)}, "
                f"label={tuple(label_tensor.shape)}"
            )
        if not torch.isfinite(input_tensor).all():
            raise FloatingPointError(f"Non-finite input after preprocessing: {record.stem}")

        result: Dict[str, object] = {"image": input_tensor, "label": label_tensor}
        if self.return_metadata:
            result.update(
                {
                    "stem": record.stem,
                    "dtm_available": self.dtm_available_mask[index],
                    "image_path": record.image_path,
                    "mask_path": record.mask_path,
                    "dtm_path": record.dtm_path,
                }
            )
        return result

    @staticmethod
    def rgb_loader(path: str) -> Image.Image:
        return load_rgb(path)

    @staticmethod
    def label_loader(path: str) -> Image.Image:
        return load_mask(path)

    @staticmethod
    def dtm_loader(
        path: str,
        config: Optional[DTMNormalizationConfig] = None,
    ) -> np.ndarray:
        raw, _ = read_dtm_masked(path)
        return normalize_dtm(raw, config or DTMNormalizationConfig())

    def preprocessing_config(self) -> Dict[str, object]:
        return {
            "trainsize": self.trainsize,
            "rgb_mean": list(IMAGENET_MEAN),
            "rgb_std": list(IMAGENET_STD),
            "dtm": self.dtm_config.to_dict(),
            "dtm_scale": self.dtm_scale,
            "dtm_availability": self.dtm_availability,
            "availability_seed": self.availability_seed,
            "mask_threshold": self.mask_threshold,
        }


class TestDataset(Dataset):
    """Inference dataset; ground truth is optional and output size comes from RGB."""

    def __init__(
        self,
        image_root: str,
        dtm_root: str,
        testsize: int,
        mask_root: Optional[str] = None,
        dtm_scale: float = 1.0,
        dtm_availability: float = 1.0,
        availability_seed: int = 42,
        dtm_norm: str = "per_tile_zscore",
        dtm_divisor: float = 65535.0,
        dtm_global_mean: Optional[float] = None,
        dtm_global_std: Optional[float] = None,
        dtm_missing_value: float = 0.0,
        strict_matching: bool = True,
    ) -> None:
        super().__init__()
        if int(testsize) <= 0:
            raise ValueError("testsize must be > 0")
        self.records, self.pairing_report = build_sample_records(
            image_root=image_root,
            mask_root=mask_root,
            dtm_root=dtm_root,
            strict=strict_matching,
        )
        self.testsize = int(testsize)
        self.size = (self.testsize, self.testsize)
        self.dtm_scale = float(dtm_scale)
        self.dtm_availability = float(dtm_availability)
        self.availability_seed = int(availability_seed)
        self.dtm_config = DTMNormalizationConfig(
            mode=dtm_norm,
            divisor=float(dtm_divisor),
            global_mean=dtm_global_mean,
            global_std=dtm_global_std,
            missing_value=float(dtm_missing_value),
        )
        self.dtm_config.validate()
        self.stems = [record.stem for record in self.records]
        self.images = [record.image_path for record in self.records]
        self.gts = [record.mask_path for record in self.records]
        self.dtms = [record.dtm_path for record in self.records]
        self.dtm_available_mask = deterministic_availability_mask(
            self.stems, self.dtm_availability, self.availability_seed
        )
        self.index = 0  # compatibility with the repository's old test loop

        print(f"[TestDataset] matched={len(self.records)} / RGB={self.pairing_report.image_count}")
        print(
            f"[TestDataset] DTM availability={sum(self.dtm_available_mask)}/{len(self.records)}; "
            f"scale={self.dtm_scale}; norm={self.dtm_config.mode}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        image = load_rgb(record.image_path)
        original_size = (image.height, image.width)
        image_resized = TF.resize(
            image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        rgb_tensor = TF.normalize(TF.to_tensor(image_resized), IMAGENET_MEAN, IMAGENET_STD)

        if self.dtm_available_mask[index]:
            raw, _ = read_dtm_masked(record.dtm_path)
            dtm = normalize_dtm(raw, self.dtm_config)
            dtm = resize_dtm_nan_safe(dtm, self.size)
            dtm = np.nan_to_num(
                dtm,
                nan=self.dtm_config.missing_value,
                posinf=self.dtm_config.missing_value,
                neginf=self.dtm_config.missing_value,
            ).astype(np.float32)
        else:
            dtm = np.full(self.size, self.dtm_config.missing_value, dtype=np.float32)

        dtm_tensor = torch.from_numpy(dtm)[None] * self.dtm_scale
        input_tensor = torch.cat((rgb_tensor, dtm_tensor), dim=0)
        if not torch.isfinite(input_tensor).all():
            raise FloatingPointError(f"Non-finite inference input: {record.stem}")

        gt = None
        if record.mask_path is not None:
            gt = np.asarray(load_mask(record.mask_path))

        return {
            "image": input_tensor,
            "gt": gt,
            "name": Path(record.image_path).name,
            "stem": record.stem,
            "original_size": original_size,
            "dtm_available": self.dtm_available_mask[index],
        }

    def load_data(self):
        """Backward-compatible stateful API used by the old ``test.py``."""
        if self.index >= len(self):
            raise IndexError("TestDataset index is out of range")
        sample = self[self.index]
        self.index += 1
        gt = sample["gt"] if sample["gt"] is not None else np.empty((0, 0), dtype=np.uint8)
        return sample["image"].unsqueeze(0), gt, sample["name"]
