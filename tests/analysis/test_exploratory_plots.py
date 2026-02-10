from src.exploratory_plots import validate_metrics, make_distribution_plot, plot_distributions
from unittest.mock import patch
import pytest

# Distribution plot unit tests.

# Test ValueError raised when metrics are missing.
def test_plot_distributions_missing_metric(sample_metrics_df):
    with pytest.raises(ValueError):
        validate_metrics(sample_metrics_df, ["not_real"])

# Test plot returns figure.
def test_make_distribution_plot_returns_fig(sample_metrics_df):
    fig, ax = make_distribution_plot(sample_metrics_df, "mcc")

    assert fig is not None
    assert ax is not None

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

