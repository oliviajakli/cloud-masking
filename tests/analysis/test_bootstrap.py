from src.analysis.bootstrap import (bootstrap_scene_tiles, list_scenes, compute_tile_mccs_all, 
    mcc_per_tile, safe_mcc, summarize, tile_array, two_level_bootstrap)
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# I/O heavy functions (mock or partial testing).
def test_list_scenes(tmp_path):
    gt = tmp_path / "gt"
    alg = tmp_path / "alg"
    gt.mkdir()
    alg.mkdir()

    (gt / "a.tif").touch()
    (gt / "b.tif").touch()
    (alg / "b.tif").touch()
    (alg / "c.tif").touch()

    scenes = list_scenes(gt, alg)

    assert scenes == ["b.tif"]

# Test for shape mismatch between GT and algorithm tiles.
def test_compute_tile_mccs_shape_mismatch(monkeypatch):
    def fake_read(path):
        if "gt" in str(path):
            return np.zeros((4, 4))
        return np.zeros((5, 5))

    monkeypatch.setattr("src.bootstrap.read_raster", fake_read)

    with pytest.raises(ValueError):
        compute_tile_mccs_all(
            gt_folder=Path("gt"),
            alg_folder=Path("alg"),
            scenes=["a.tif"],
            tile=2,
        )

# Pure logic functions (strict unit tests).

# Test perfect division.
def test_tile_array_exact_division():
    arr = np.arange(16).reshape(4, 4)
    tiles = tile_array(arr, 2, 2)

    assert len(tiles) == 4
    for t in tiles:
        assert t.shape == (2, 2)

# Test remainder edges are discarded.
def test_tile_array_discards_partial_tiles():
    arr = np.arange(15).reshape(3, 5)
    tiles = tile_array(arr, 2, 2)

    # Only full tiles count
    assert all(t.shape == (2, 2) for t in tiles)

# Test correct MCC computation.
def test_safe_mcc_valid_case():
    y_true = np.array([[1, 0], [1, 0]])
    y_pred = np.array([[1, 0], [0, 1]])

    mcc = safe_mcc(y_true, y_pred)

    assert -1 <= mcc <= 1

# Test: Degenerate case returns NaN.
def test_safe_mcc_single_class():
    y_true = np.zeros((4, 4))
    y_pred = np.zeros((4, 4))

    mcc = safe_mcc(y_true, y_pred)

    assert np.isnan(mcc)

def test_mcc_per_tile_basic():
    gt = np.zeros((4, 4))
    pred = np.zeros((4, 4))

    result = mcc_per_tile(gt, pred, tile=2)

    assert result.shape[0] == 4
    assert np.isnan(result).all()

def test_bootstrap_scene_tiles_all_nan():
    arr = np.array([np.nan, np.nan])
    boots = bootstrap_scene_tiles(arr, B=10)

    assert len(boots) == 10
    assert np.isnan(boots).all()

def test_bootstrap_scene_tiles_valid():
    np.random.seed(0)
    arr = np.array([0.5, 0.6, 0.7])
    boots = bootstrap_scene_tiles(arr, B=100)

    assert len(boots) == 100
    assert boots.min() >= 0.5
    assert boots.max() <= 0.7

# Test summary shape and bounds.
def test_summarize_basic():
    df = pd.DataFrame({
        "alg1": np.random.rand(100),
        "alg2": np.random.rand(100),
    })

    summary = summarize(df)

    assert summary.shape[0] == 2
    assert set(summary.columns) == {"metric", "median", "ci_lower", "ci_upper"}

    for _, row in summary.iterrows():
        assert row["ci_lower"] <= row["median"] <= row["ci_upper"]

# Test statistical engines.
def make_fake_scene_data():
    return {
        "alg1": {
            "scene1": np.array([0.5, 0.6]),
            "scene2": np.array([0.4, 0.7]),
        },
        "alg2": {
            "scene1": np.array([0.2, 0.3]),
            "scene2": np.array([0.1, 0.2]),
        },
    }

def test_two_level_bootstrap_structure():
    data = make_fake_scene_data()

    df_metrics, df_diffs = two_level_bootstrap(
        data,
        B_scene=10,
        B_global=20,
        seed=42,
    )

    assert df_metrics.shape == (20, 2)
    assert df_diffs.shape == (20, 1)
    assert "alg1_minus_alg2" in df_diffs.columns

def test_two_level_bootstrap_reproducible():
    data = make_fake_scene_data()

    df1, _ = two_level_bootstrap(data, 10, 20, seed=123)
    df2, _ = two_level_bootstrap(data, 10, 20, seed=123)

    pd.testing.assert_frame_equal(df1, df2)

def test_two_level_bootstrap_diff_direction():
    data = make_fake_scene_data()

    df_metrics, df_diffs = two_level_bootstrap(
        data,
        B_scene=10,
        B_global=20,
        seed=1,
    )

    assert (df_diffs["alg1_minus_alg2"] >= -1).all()
