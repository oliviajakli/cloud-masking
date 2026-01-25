import pandas as pd     # type: ignore
from scipy.stats import shapiro
from pathlib import Path
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns   # type: ignore
import os

from src.utils.io import save_csv
from src.utils.plotting import save_figure


def compute_pairwise_differences(df: pd.DataFrame, output_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute pairwise differences in MCC between algorithms.
    Args:
        df (pd.DataFrame): DataFrame with columns 'scene_id', 'algorithm', 'mcc'.
        output_dir (Path): Directory to save output CSV files.
    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: Pairwise differences in MCC:
            (hybrid - s2cloudless, hybrid - cloudscoreplus, s2cloudless - cloudscoreplus)
    """
    # Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
    mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')
    # Compute pairwise differences. Cloud Score+ is at index 0.
    # Hybrid method is at index 1. s2cloudless at index 2.
    diff_hy_s2 = mcc_wide.iloc[:, 1] - mcc_wide.iloc[:, 2]
    diff_hy_cs = mcc_wide.iloc[:, 1] - mcc_wide.iloc[:, 0]
    diff_s2_cs = mcc_wide.iloc[:, 2] - mcc_wide.iloc[:, 0]
    # Save differences to output directory.
    output_path = output_dir / "pairwise_mcc_differences.csv"
    diff_df = pd.DataFrame({
        'scene_id': mcc_wide.index,
        'diff_hybrid_s2cloudless': diff_hy_s2,
        'diff_hybrid_cloudscoreplus': diff_hy_cs,
        'diff_s2cloudless_cloudscoreplus': diff_s2_cs
    })
    save_csv(diff_df, output_path)
    return diff_hy_s2, diff_hy_cs, diff_s2_cs

def test_normality(df: pd.DataFrame, pairs: list, output_dir: Path) -> None:
    """Test normality of pairwise differences in MCC using Shapiro–Wilk test.
    Args:
        df (pd.DataFrame): DataFrame with columns 'scene_id', 'algorithm', 'mcc'.
        pairs (list): List of algorithm pairs to compare.
        output_dir (Path): Directory to save output CSV files and figures.
    Returns:
        None
    """
    diff_hy_s2, diff_hy_cs, diff_s2_cs = compute_pairwise_differences(df, output_dir)

    # Shapiro–Wilk normality test for pairwise differences.
    for pair, diff in zip(pairs, [diff_hy_s2, diff_hy_cs, diff_s2_cs]):
        stat, p = shapiro(diff)
        # Create label from pair tuple/list.
        label = f"{pair[0]} - {pair[1]}" if isinstance(pair, (list, tuple)) else pair
        # Save normality test results as CSV.
        output_path = output_dir / f"shapiro_wilk_{label.replace(' ', '_').replace('-', 'vs')}.csv"
        result_df = pd.DataFrame({'algorithm_pair': [label], 'statistic': [stat], 'p_value': [p]})
        save_csv(result_df, output_path)
        print(f"{label}: stat = {stat:.4f}, p = {p:.4f}")

    diff_series_list = [
        ("hybrid - s2cloudless (MCC)", diff_hy_s2),
        ("hybrid - cloudscoreplus (MCC)", diff_hy_cs),
        ("s2cloudless - cloudscoreplus (MCC)", diff_s2_cs),
    ]

    for title_label, diff_series in diff_series_list:
        plt.figure(figsize=(7, 5))

        sns.histplot(
            x=diff_series,  # Pass the Series directly to x
            bins=15,
            stat='density',
            element='step',
            color='steelblue', # Assign a color as hue is removed
            alpha=0.3
        )

        sns.kdeplot(
            x=diff_series, # Pass the Series directly to x
            color='steelblue', # Assign a color as hue is removed
            linewidth=2
        )

        fig_path = os.path.join(
            output_dir, 
            f'histogram_kde_mcc_diff_{title_label.replace(" ", "_")
            .replace("-", "vs").replace("(", "").replace(")", "")}.png'
            )

        plt.title(f'Distribution of MCC Differences: {title_label}')
        plt.xlabel('MCC Difference')
        plt.ylabel('Density')
        save_figure(plt.gcf(), Path(fig_path))