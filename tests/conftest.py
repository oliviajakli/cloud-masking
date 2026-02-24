import pytest
import numpy as np
import rasterio     # type: ignore
from rasterio.transform import from_origin  # type: ignore
import pandas as pd

from src.utils.validator import DataValidator     # type: ignore

@pytest.fixture
def sample_metrics_df():
    return pd.DataFrame({
        "scene_id": [1, 2, 3],
        "algorithm": ["A", "B", "C"],
        "f1_score": [0.9, 0.85, 0.75],
        "iou": [0.5, 0.35, 0.63],
        "mcc": [0.8, 0.7, 0.6],
        "cloud_fraction": [0.08, 0.98, 0.22]
    })


@pytest.fixture
def small_binary_raster(tmp_path):
    path = tmp_path / "mask.tif"
    data = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=data.dtype,
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data, 1)

    return path


@pytest.fixture
def write_raster():
    def _write(arr, path):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            transform=from_origin(0, 0, 1, 1),
        ) as dst:
            dst.write(arr, 1)
    return _write

@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "algorithm": ["A", "B"],
        "scene_id": [1, 1],
        "precision": [0.8, 0.9],
        "recall": [0.7, 0.85],
    })

@pytest.fixture
def validator():
    return DataValidator(
        required_columns={"algorithm", "scene_id", "precision", "recall"},
        metric_columns={"precision", "recall"},
        expected_algorithms={"A", "B"},
        value_constraints={
            "precision": lambda s: s.between(0, 1),
            "recall": lambda s: s.between(0, 1),
        }
    )