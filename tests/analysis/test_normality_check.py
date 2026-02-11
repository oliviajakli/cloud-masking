import pandas as pd
from src.normality_check import compute_pairwise_differences, plot_normality, shapiro_wilk_test
import pytest

def test_compute_pairwise_differences_correct(tmp_path):
    df = pd.DataFrame({
        "scene_id": [1, 1, 1, 2, 2, 2],
        "algorithm": [
            "cloudscoreplus", "hybrid", "s2cloudless",
            "cloudscoreplus", "hybrid", "s2cloudless"
        ],
        "mcc": [0.7, 0.8, 0.6, 0.75, 0.85, 0.65],
    })

    diff_hy_s2, diff_hy_cs, diff_s2_cs = compute_pairwise_differences(
        df=df,
        output_dir=tmp_path,
    )

    # Length equals number of scenes
    assert len(diff_hy_s2) == 2

    # Directionality checks
    assert (diff_hy_s2 > 0).all()
    assert (diff_hy_cs > 0).all()
    assert (diff_s2_cs < 0).all()

    # CSV saved
    assert (tmp_path / "pairwise_mcc_differences.csv").exists()

def test_compute_pairwise_differences_missing_algorithm(tmp_path):
    df = pd.DataFrame({
        "scene_id": [1, 1],
        "algorithm": ["hybrid", "s2cloudless"],
        "mcc": [0.8, 0.6],
    })

    with pytest.raises(ValueError):
        compute_pairwise_differences(df, tmp_path)

def test_shapiro_wilk_test_structure():
    diffs = [
        pd.Series([0.1, 0.2, 0.15]),
        pd.Series([-0.05, 0.0, 0.02]),
        pd.Series([0.3, 0.25, 0.28]),
    ]
    pairs = [
        "hybrid - s2cloudless",
        "hybrid - cloudscoreplus",
        "s2cloudless - cloudscoreplus",
    ]

    result = shapiro_wilk_test(
        pairs=pairs,
        diff_pair1=diffs[0],
        diff_pair2=diffs[1],
        diff_pair3=diffs[2],
    )

    assert result.shape == (3, 3)
    assert set(result.columns) == {"algorithm_pair", "statistic", "p_value"}
    assert ((result["p_value"] >= 0) & (result["p_value"] <= 1)).all()

def test_shapiro_wilk_test_mismatched_lengths():
    with pytest.raises(ValueError):
        shapiro_wilk_test(
            pairs=["A", "B"],
            diff_pair1=pd.Series([1, 2]),
            diff_pair2=pd.Series([1, 2]),
            diff_pair3=pd.Series([1, 2])
        )

def test_plot_normality_creates_files(tmp_path):
    diffs = [
        pd.Series([0.1, 0.2, 0.15]),
        pd.Series([-0.05, 0.0, 0.02]),
        pd.Series([0.3, 0.25, 0.28]),
    ]

    plot_normality(
        diff_pair1=diffs[0],
        diff_pair2=diffs[1],
        diff_pair3=diffs[2],
        output_dir=tmp_path,
    )

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 3