from unittest.mock import patch

import numpy as np
import pytest

from src.analysis.evaluation import compute_metrics, plot_confusion_matrix


# Unit tests for compute_metrics.
def test_compute_metrics_columns(tmp_path, write_raster):
    masks_dir = tmp_path / "masks"
    ref_dir = masks_dir / "reference"
    alg_dir = masks_dir / "alg_a"

    ref_dir.mkdir(parents=True)
    alg_dir.mkdir(parents=True)

    data = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    write_raster(data, ref_dir / "scene1.tif")
    write_raster(data, alg_dir / "scene1.tif")

    df = compute_metrics(masks_dir)

    expected_cols = {
        "scene_id", "algorithm", "TP", "TN", "FP", "FN",
        "balanced_accuracy", "precision", "recall",
        "f1_score", "iou", "mcc",
        "FPR", "FNR", "cloud_fraction"
    }

    assert set(df.columns) == expected_cols


def test_compute_metrics_confusion_counts(tmp_path, write_raster):
    masks_dir = tmp_path / "masks"
    ref_dir = masks_dir / "reference"
    alg_dir = masks_dir / "alg_a"
    ref_dir.mkdir(parents=True)
    alg_dir.mkdir(parents=True)

    ref = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    pred = np.array([[0, 1], [0, 0]], dtype=np.uint8)

    write_raster(ref, ref_dir / "scene1.tif")
    write_raster(pred, alg_dir / "scene1.tif")

    df = compute_metrics(masks_dir)

    row = df.iloc[0]
    assert row.TP == 1
    assert row.TN == 2
    assert row.FP == 0
    assert row.FN == 1

def test_cloud_fraction(tmp_path, write_raster):
    # TP=1, FP=1, TN=1, FN=1 → cloud_fraction = 0.5
    masks_dir = tmp_path / "masks"
    ref_dir = masks_dir / "reference"
    alg_dir = masks_dir / "alg_a"
    ref_dir.mkdir(parents=True)
    alg_dir.mkdir(parents=True)

    ref = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    write_raster(ref, ref_dir / "scene1.tif")
    write_raster(pred, alg_dir / "scene1.tif")

    df = compute_metrics(masks_dir)
    row = df.iloc[0]
    assert np.isclose(row.cloud_fraction, 0.5)

def test_non_tif_files_skipped(tmp_path):
    masks_dir = tmp_path / "masks"
    ref_dir = masks_dir / "reference"
    alg_dir = masks_dir / "alg_a"
    ref_dir.mkdir(parents=True)
    alg_dir.mkdir(parents=True)

    (alg_dir / "readme.txt").write_text("ignore me")
    (ref_dir / "readme.txt").write_text("ignore me too")

    df = compute_metrics(masks_dir)

    assert df.empty

def test_empty_algorithm_folder_raises(tmp_path):
    masks_dir = tmp_path / "masks"
    (masks_dir / "reference").mkdir(parents=True)
    (masks_dir / "alg_a").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        compute_metrics(masks_dir)

# Unit tests for plot_confusion_matrix.
def test_plot_confusion_matrix_invalid_shape():
    cm = np.zeros((3, 3))
    with pytest.raises(ValueError):
        plot_confusion_matrix(cm, "Invalid")

def test_plot_confusion_matrix_empty():
    cm = np.zeros((2, 2))
    with pytest.raises(ValueError):
        plot_confusion_matrix(cm, "Empty")

@patch("src.evaluation.save_figure")
def test_plot_confusion_matrix_saves(mock_save):
    cm = np.array([[5, 1], [2, 7]])
    plot_confusion_matrix(cm, "Test Matrix")

    assert mock_save.called
    args, kwargs = mock_save.call_args
    assert "test_matrix.png" in str(args[1])
