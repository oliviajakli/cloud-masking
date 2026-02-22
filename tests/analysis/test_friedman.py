from src.analysis.friedman import run_friedman_test
import pandas as pd
import pytest

def test_run_friedman_test_basic():
    df = pd.DataFrame({
        "scene_id": [1, 1, 1, 2, 2, 2],
        "algorithm": ["A", "B", "C", "A", "B", "C"],
        "mcc": [0.7, 0.6, 0.8, 0.75, 0.65, 0.85],
    })

    result = run_friedman_test(df)

    assert result.shape == (1, 2)
    assert set(result.columns) == {"statistic", "p_value"}
    assert 0 <= result["p_value"].iloc[0] <= 1

def test_run_friedman_test_requires_3_algorithms():
    df = pd.DataFrame({
        "scene_id": [1, 1],
        "algorithm": ["A", "B"],
        "mcc": [0.7, 0.6],
    })

    with pytest.raises(ValueError):
        run_friedman_test(df)

def test_run_friedman_test_missing_scene_alignment():
    df = pd.DataFrame({
        "scene_id": [1, 2, 1, 2],
        "algorithm": ["A", "A", "B", "B"],
        "mcc": [0.7, 0.75, 0.6, None],
    })

    with pytest.raises(ValueError):
        run_friedman_test(df)
