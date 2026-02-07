import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest
import pandas as pd     # type: ignore

@pytest.fixture
def sample_metrics_df():
    return pd.DataFrame({
        "scene_id": [1, 2, 3],
        "algorithm": ["A", "B", "C"],
        "mcc": [0.8, 0.7, 0.6],
        "f1": [0.9, 0.85, 0.75],
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
