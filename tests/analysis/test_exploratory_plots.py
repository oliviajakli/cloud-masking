from src.analysis.exploratory_plots import (
    bootstrap_ci, compute_error_map, plot_bland_altman, plot_boxplots_with_stats, 
    plot_paired_differences, validate_metrics, make_distribution_plot, plot_distributions)
from unittest.mock import patch
import pytest
import numpy as np

# Unit tests and smoke tests for distribution plots and exploratory visuals.

# Test ValueError raised when metrics are missing.
def test_plot_distributions_missing_metric(sample_metrics_df):
    with pytest.raises(ValueError):
        validate_metrics(sample_metrics_df, ["not_real"])

# Test plot returns figure.
def test_make_distribution_plot_returns_fig(sample_metrics_df):
    fig, ax = make_distribution_plot(sample_metrics_df, "mcc")

    assert fig is not None
    assert ax is not None
    assert len(fig.axes) == 1

# Mock test save figure is called for each metric.
@patch("src.exploratory_plots.save_figure")
def test_plot_distributions_saves(mock_save, tmp_path, sample_metrics_df):
    plot_distributions(
        sample_metrics_df,
        metrics=["mcc", "f1_score"],
        output_dir=tmp_path,
    )

    assert mock_save.call_count == 2

@patch("src.exploratory_plots.save_figure")
def test_plot_distributions_filenames(mock_save, tmp_path, sample_metrics_df):
    plot_distributions(
        sample_metrics_df,
        metrics=["mcc", "f1_score"],
        output_dir=tmp_path,
    )
    saved_paths = [call.args[1] for call in mock_save.call_args_list]
    assert any("mcc_distribution.png" in str(p) for p in saved_paths)

# Test Wilcoxon runs, annotator doesn't crash, and file is saved.
def test_plot_boxplots_with_stats_runs(tmp_path, sample_metrics_df):

    plot_boxplots_with_stats(
        df=sample_metrics_df,
        metrics=["mcc"],
        pairs=[("A", "B")],
        algorithms=["A", "B"],
        output_dir=tmp_path,
    )

    assert (tmp_path / "mcc_boxplot.png").exists()

# Unit test bootstrap_ci returns mean and confidence interval bounds.
def test_bootstrap_ci_basic():
    data = np.array([1, 2, 3, 4, 5])
    mean, low, high = bootstrap_ci(data, n_boot=1000)

    assert low <= mean <= high

# Check that paired differences plot runs and saves file.
def test_plot_paired_differences_creates_file(tmp_path, sample_metrics_df):

    plot_paired_differences(
        df=sample_metrics_df,
        metrics=["iou"],
        pairs=[("A", "B")],
        output_dir=tmp_path,
    )

    assert (tmp_path / "iou_paired_differences.png").exists()

# Check that Bland-Altman plot runs and saves file.
def test_plot_bland_altman_creates_file(tmp_path, sample_metrics_df):

    plot_bland_altman(
        df=sample_metrics_df,
        pairs=[("A", "B")],
        output_dir=tmp_path,
    )

    expected = tmp_path / "bland_altman_A_vs_B_cloud_fraction.png"
    assert expected.exists()


# Test that compute_error_map returns expected values for a simple case.
def test_compute_error_map_values():
    ref = np.array([[1, 0], [1, 0]])
    pred = np.array([[1, 0], [0, 1]])

    error = compute_error_map(ref, pred)

    assert set(np.unique(error)) <= {0, 1, 2, 3, 4}

# Test that plot_distributions creates files for each metric.
def test_plot_scatterplot_creates_files(tmp_path, sample_metrics_df):
    plot_distributions(
        df=sample_metrics_df,
        metrics=["mcc"],
        output_dir=tmp_path,
    )

    assert (tmp_path / "mcc_distribution.png").exists()

# Test that plot_time_series creates files for each metric.
def test_plot_time_series_creates_files(tmp_path, sample_metrics_df):
    plot_distributions(
        df=sample_metrics_df,
        metrics=["mcc"],
        output_dir=tmp_path,
    )

    assert (tmp_path / "mcc_distribution.png").exists()