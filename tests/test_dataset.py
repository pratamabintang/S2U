from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin

from dataset import (
    DatasetIntegrityError,
    FullDataset,
    TestDataset as InferenceDataset,
    build_sample_records,
    deterministic_availability_mask,
    read_dtm_masked,
    resize_dtm_nan_safe,
)


def write_rgb(path: Path, shape=(8, 10)):
    array = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    array[..., 0] = 120
    array[..., 1] = 80
    Image.fromarray(array).save(path)


def write_mask(path: Path, shape=(8, 10), rgb=False):
    array = np.zeros(shape, dtype=np.uint8)
    array[2:5, 3:7] = 255
    if rgb:
        array = np.repeat(array[..., None], 3, axis=2)
    Image.fromarray(array).save(path)


def write_dtm(path: Path, array: np.ndarray, nodata=-9999.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        nodata=nodata,
        transform=from_origin(100, 200, 1, 1),
    ) as dst:
        dst.write(array.astype(np.float32), 1)


def make_roots(tmp_path: Path):
    image_root = tmp_path / "image"
    mask_root = tmp_path / "mask"
    dtm_root = tmp_path / "dtm"
    image_root.mkdir()
    mask_root.mkdir()
    dtm_root.mkdir()
    return image_root, mask_root, dtm_root


def test_case_insensitive_pairing_and_four_channel_tensor(tmp_path):
    image_root, mask_root, dtm_root = make_roots(tmp_path)
    write_rgb(image_root / "Grid_19.png")
    write_mask(mask_root / "GRID_19.PNG", rgb=True)
    dtm = np.arange(20, dtype=np.float32).reshape(4, 5)
    dtm[0, 0] = -9999
    write_dtm(dtm_root / "grid_19.TIF", dtm)

    dataset = FullDataset(
        str(image_root),
        str(mask_root),
        str(dtm_root),
        trainsize=16,
        mode="val",
        mask_threshold=127,
    )
    sample = dataset[0]
    assert sample["image"].shape == (4, 16, 16)
    assert sample["label"].shape == (1, 16, 16)
    assert torch.isfinite(sample["image"]).all()
    assert set(torch.unique(sample["label"]).tolist()) <= {0.0, 1.0}
    assert sample["stem"] == "Grid_19"


def test_validation_and_inference_preprocessing_are_identical(tmp_path):
    image_root, mask_root, dtm_root = make_roots(tmp_path)
    write_rgb(image_root / "tile.png")
    write_mask(mask_root / "tile.png")
    dtm = np.arange(80, dtype=np.float32).reshape(8, 10)
    dtm[0, 0] = -9999.0
    write_dtm(dtm_root / "tile.tif", dtm)

    common = dict(
        dtm_scale=0.75,
        dtm_availability=1.0,
        availability_seed=7,
        dtm_norm="per_tile_zscore",
        dtm_divisor=65535.0,
        dtm_missing_value=0.0,
        strict_matching=True,
    )
    validation = FullDataset(
        str(image_root),
        str(mask_root),
        str(dtm_root),
        trainsize=16,
        mode="val",
        return_metadata=True,
        **common,
    )
    inference = InferenceDataset(
        str(image_root),
        str(dtm_root),
        mask_root=str(mask_root),
        testsize=16,
        **common,
    )

    val_sample = validation[0]
    test_sample = inference[0]
    assert torch.equal(val_sample["image"], test_sample["image"])
    assert test_sample["original_size"] == (8, 10)
    assert test_sample["gt"].shape == (8, 10)


def test_strict_pairing_fails_instead_of_silently_dropping(tmp_path):
    image_root, mask_root, dtm_root = make_roots(tmp_path)
    write_rgb(image_root / "a.png")
    write_mask(mask_root / "a.png")
    with pytest.raises(DatasetIntegrityError, match="missing DTM"):
        build_sample_records(image_root, dtm_root, mask_root, strict=True)


def test_duplicate_casefolded_stems_are_rejected(tmp_path):
    image_root, mask_root, dtm_root = make_roots(tmp_path)
    write_rgb(image_root / "A.png")
    write_rgb(image_root / "a.jpg")
    write_mask(mask_root / "a.png")
    write_dtm(dtm_root / "a.tif", np.ones((8, 10), dtype=np.float32))
    with pytest.raises(DatasetIntegrityError, match="Ambiguous RGB"):
        build_sample_records(image_root, dtm_root, mask_root)


def test_nan_safe_resize_does_not_smear_nonfinite_values():
    array = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    resized = resize_dtm_nan_safe(array, (8, 8))
    assert resized.shape == (8, 8)
    assert np.isfinite(resized).sum() > 0
    # The finite region remains bounded by finite source values.
    finite = resized[np.isfinite(resized)]
    assert finite.min() >= 1.0 - 1e-5
    assert finite.max() <= 5.0 + 1e-5


def test_rasterio_masked_read_honours_nodata(tmp_path):
    path = tmp_path / "dtm.tif"
    array = np.array([[1, -9999], [3, 4]], dtype=np.float32)
    write_dtm(path, array)
    read, metadata = read_dtm_masked(path)
    assert np.isnan(read[0, 1])
    assert metadata["shape"] == (2, 2)

    nan_path = tmp_path / "dtm_nan_nodata.tif"
    nan_array = np.array([[1, np.nan], [3, 4]], dtype=np.float32)
    write_dtm(nan_path, nan_array, nodata=np.nan)
    nan_read, nan_metadata = read_dtm_masked(nan_path)
    assert np.isnan(nan_read[0, 1])
    assert nan_metadata["nodata"] is None
    assert nan_metadata["nodata_is_nan"] is True


def test_rasterio_masked_read_supports_uint16_dtm(tmp_path):
    path = tmp_path / "dtm_uint16.tif"
    array = np.array([[1000, 65535], [3000, 4000]], dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint16",
        nodata=65535,
        transform=from_origin(100, 200, 1, 1),
    ) as dst:
        dst.write(array, 1)

    read, metadata = read_dtm_masked(path)
    assert read.dtype == np.float32
    assert read[0, 0] == 1000.0
    assert np.isnan(read[0, 1])
    assert metadata["dtype"] == "uint16"
    assert metadata["nodata"] == 65535.0


def test_availability_is_exact_reproducible_and_nested():
    stems = [f"tile_{index:03d}" for index in range(20)]
    mask25 = deterministic_availability_mask(stems, 0.25, 42)
    mask50 = deterministic_availability_mask(stems, 0.50, 42)
    assert sum(mask25) == 5
    assert sum(mask50) == 10
    assert all((not selected25) or selected50 for selected25, selected50 in zip(mask25, mask50))
    assert mask25 == deterministic_availability_mask(stems, 0.25, 42)
