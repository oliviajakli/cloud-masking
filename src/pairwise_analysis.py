import pandas as pd
from scipy.stats import shapiro
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.utils.io import save_csv
from src.utils.plotting import save_figure


def compute_pairwise_differences(df, output_dir):
    # Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
    mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')
    # Compute pairwise differences. Cloud Score+ is at index 0.
    # Hybrid method is at index 1. s2cloudless at index 2.
    diff_hy_s2 = mcc_wide.iloc[:, 1] - mcc_wide.iloc[:, 2]
    diff_hy_cs = mcc_wide.iloc[:, 1] - mcc_wide.iloc[:, 0]
    diff_s2_cs = mcc_wide.iloc[:, 2] - mcc_wide.iloc[:, 0]
    # Save differences to output directory
    output_path = output_dir / "pairwise_mcc_differences.csv"
    diff_df = pd.DataFrame({
        'scene_id': mcc_wide.index,
        'diff_hybrid_s2cloudless': diff_hy_s2,
        'diff_hybrid_cloudscoreplus': diff_hy_cs,
        'diff_s2cloudless_cloudscoreplus': diff_s2_cs
    })
    save_csv(diff_df, output_path)
    return diff_hy_s2, diff_hy_cs, diff_s2_cs

def test_normality(df, pairs, output_dir):
    diff_hy_s2, diff_hy_cs, diff_s2_cs = compute_pairwise_differences(df, output_dir)

    # Shapiro–Wilk normality test for pairwise differences.
    for label, diff in zip(pairs, [diff_hy_s2, diff_hy_cs, diff_s2_cs]):
        stat, p = shapiro(diff)
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