import pandas as pd  # type: ignore
import pytest

from src.descriptive_stats import compute_descriptive_stats

# Unit tests for compute_descriptive_stats.

# Test correct schema of output DataFrame.
def test_descriptive_stats_schema(sample_metrics_df: pd.DataFrame):
    metrics = ["mcc", "f1_score"]
    result = compute_descriptive_stats(sample_metrics_df, metrics)

    assert set(result.columns) == {
        "algorithm", "metric", "median", "std"
    }

# Test correct median computation for a simple case.
def test_descriptive_stats_median(sample_metrics_df: pd.DataFrame):
    result = compute_descriptive_stats(sample_metrics_df, ["mcc"])

    row = result[
        (result.algorithm == "A") & (result.metric == "mcc")].iloc[0]

    expected_median = sample_metrics_df[
        sample_metrics_df.algorithm == "A"]["mcc"].median()
    
    assert row["median"] == expected_median

# Test that missing metric raises ValueError.
def test_descriptive_stats_missing_metric(sample_metrics_df):
    with pytest.raises(ValueError):
        compute_descriptive_stats(sample_metrics_df, ["not_a_metric"])
