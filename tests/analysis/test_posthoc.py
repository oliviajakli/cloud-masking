from src.posthoc_tests import run_posthoc_wilcoxon, effect_size_cliffs_delta, bootstrap_cliffs_delta
import pandas as pd
import numpy as np
import pytest

def test_run_posthoc_wilcoxon_structure():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [1, 2, 2, 3],
        "C": [4, 3, 2, 1],
    })

    pairs = [("A", "B"), ("A", "C"), ("B", "C")]

    result = run_posthoc_wilcoxon(df, pairs)

    assert result.shape[0] == 3
    assert set(result.columns) == {
        "algorithm_pair",
        "uncorrected_p_value",
        "corrected_p_value",
        "reject_null",
    }

    assert ((result["uncorrected_p_value"] >= 0) &
            (result["uncorrected_p_value"] <= 1)).all()

    assert ((result["corrected_p_value"] >= 0) &
            (result["corrected_p_value"] <= 1)).all()

    assert result["reject_null"].dtype == bool

def test_wilcoxon_corrected_p_not_smaller():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [1, 2, 2, 3],
        "C": [4, 3, 2, 1],
    })

    pairs = [("A", "B"), ("A", "C"), ("B", "C")]

    result = run_posthoc_wilcoxon(df, pairs)

    assert (result["corrected_p_value"] >=
            result["uncorrected_p_value"]).all()

def test_wilcoxon_missing_column():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
    })

    pairs = [("A", "C")]

    with pytest.raises(KeyError):
        run_posthoc_wilcoxon(df, pairs)

def test_wilcoxon_all_equal_values():
    df = pd.DataFrame({
        "A": [1, 1, 1, 1],
        "B": [1, 1, 1, 1],
    })

    pairs = [("A", "B")]

    with pytest.raises(ValueError):
        run_posthoc_wilcoxon(df, pairs)

# Tests for effect_size_cliffs_delta.
# Basic structure test.
def test_effect_size_cliffs_delta_structure():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [3, 4, 5],
    })

    pairs = [("A", "B")]

    result = effect_size_cliffs_delta(df, pairs)

    assert isinstance(result, list)
    assert len(result) == 1

    r = result[0]

    assert set(r.keys()) == {
        "algorithm_a",
        "algorithm_b",
        "delta",
        "effect_size",
        "favors",
    }

    assert -1 <= r["delta"] <= 1

# Favoring direction test.
def test_cliffs_delta_direction():
    df = pd.DataFrame({
        "A": [10, 10, 10],
        "B": [1, 1, 1],
    })

    pairs = [("A", "B")]

    result = effect_size_cliffs_delta(df, pairs)[0]

    assert result["delta"] > 0
    assert result["favors"] == "A"

# Neutral case test.
def test_cliffs_delta_neutral():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],
    })

    result = effect_size_cliffs_delta(df, [("A", "B")])[0]

    assert abs(result["delta"]) < 1e-8
    assert result["favors"] == "Neither"

# Tests for bootstrap cliff's delta.
# Valid CI bounds test.
def test_bootstrap_cliffs_delta_bounds():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 3, 4, 5, 6])

    lower, upper = bootstrap_cliffs_delta(x, y, n_boot=1000)

    assert lower <= upper
    assert -1 <= lower <= 1
    assert -1 <= upper <= 1

# Deterministic output test.
def test_bootstrap_cliffs_delta_reproducible():
    np.random.seed(42)

    x = pd.Series([1, 2, 3, 4])
    y = pd.Series([2, 3, 4, 5])

    lower1, upper1 = bootstrap_cliffs_delta(x, y, n_boot=500)

    np.random.seed(42)

    lower2, upper2 = bootstrap_cliffs_delta(x, y, n_boot=500)

    assert lower1 == lower2
    assert upper1 == upper2

# Test for error on unequal length inputs.
def test_bootstrap_requires_equal_length():
    x = pd.Series([1, 2, 3])
    y = pd.Series([1, 2])

    with pytest.raises(ValueError):
        bootstrap_cliffs_delta(x, y)