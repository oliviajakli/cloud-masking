from src.analysis.directional_error import (compute_precision_recall_diff, bootstrap_median_ci, 
    plot_directional_bias, summary_table, wilcoxon_vs_zero)
import pandas as pd
import numpy as np
import pytest

def test_compute_precision_recall_diff_basic():
    df = pd.DataFrame({
        "precision": [0.8, 0.6],
        "recall": [0.5, 0.6],
    })

    result = compute_precision_recall_diff(df)

    assert "pr_diff" in result.columns
    assert result["pr_diff"].tolist() == [0.3, 0.0]

def test_compute_precision_recall_diff_no_mutation():
    df = pd.DataFrame({
        "precision": [0.8],
        "recall": [0.5],
    })

    _ = compute_precision_recall_diff(df)

    assert "pr_diff" not in df.columns

def test_compute_precision_recall_diff_missing_column():
    df = pd.DataFrame({"precision": [0.8]})

    with pytest.raises(KeyError):
        compute_precision_recall_diff(df)

def test_bootstrap_median_ci_bounds():
    x = np.array([1, 2, 3, 4, 5])

    median, lower, upper = bootstrap_median_ci(x, n_boot=1000)

    assert lower <= median <= upper

def test_bootstrap_median_ci_reproducible():
    x = np.array([1, 2, 3, 4])

    result1 = bootstrap_median_ci(x, n_boot=500, random_state=123)
    result2 = bootstrap_median_ci(x, n_boot=500, random_state=123)

    assert result1 == result2

def test_bootstrap_median_ci_nan_handling():
    x = np.array([1, 2, np.nan, 4])

    median, lower, upper = bootstrap_median_ci(x, n_boot=200)

    assert not np.isnan(median)
    assert lower <= median <= upper

def test_bootstrap_median_ci_all_nan():
    x = np.array([np.nan, np.nan])

    median, lower, upper = bootstrap_median_ci(x, n_boot=100)

    assert np.isnan(median)
    assert np.isnan(lower)
    assert np.isnan(upper)

def test_wilcoxon_vs_zero_nonzero():
    x = np.array([1, 2, 3, 4])

    p = wilcoxon_vs_zero(x)

    assert 0 <= p <= 1

def test_wilcoxon_vs_zero_all_zero():
    x = np.zeros(5)

    p = wilcoxon_vs_zero(x)

    assert np.isnan(p)

def test_wilcoxon_vs_zero_nan_removed():
    x = np.array([1, 2, np.nan, 3])

    p = wilcoxon_vs_zero(x)

    assert 0 <= p <= 1

def test_wilcoxon_vs_zero_single_value():
    x = np.array([1])

    with pytest.raises(ValueError):
        wilcoxon_vs_zero(x)

def test_summary_table_structure():
    df = pd.DataFrame({
        "algorithm": ["A", "A", "B", "B"],
        "pr_diff": [0.2, -0.1, 0.3, 0.4],
    })

    summary = summary_table(df)

    assert set(summary.columns) == {
        "algorithm",
        "median_pr_diff",
        "ci_lower",
        "ci_upper",
        "wilcoxon_p",
        "pct_commission",
    }

    assert len(summary) == 2

def test_summary_table_pct_commission():
    df = pd.DataFrame({
        "algorithm": ["A", "A"],
        "pr_diff": [0.2, -0.1],
    })

    summary = summary_table(df)

    pct = summary.loc[summary["algorithm"] == "A", "pct_commission"].iloc[0]

    assert pct == 50.0

def test_summary_table_ci_order():
    df = pd.DataFrame({
        "algorithm": ["A"] * 10,
        "pr_diff": np.random.randn(10),
    })

    summary = summary_table(df)

    row = summary.iloc[0]

    assert row["ci_lower"] <= row["median_pr_diff"] <= row["ci_upper"]

def test_summary_table_all_zero():
    df = pd.DataFrame({
        "algorithm": ["A"] * 5,
        "pr_diff": np.zeros(5),
    })

    summary = summary_table(df)

    row = summary.iloc[0]

    assert np.isnan(row["wilcoxon_p"])

def test_plot_directional_bias_creates_file(tmp_path):
    df = pd.DataFrame({
        "algorithm": ["A", "A", "B", "B"],
        "pr_diff": [0.1, -0.1, 0.2, -0.2],
    })

    plot_directional_bias(df, tmp_path)

    output = tmp_path / "directional_bias_violinplot.png"

    assert output.exists()