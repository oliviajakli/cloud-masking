import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

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