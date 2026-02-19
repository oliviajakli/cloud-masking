from src.utils.io import save_csv
from src.utils.plotting import save_figure

import os
import logging
import pandas as pd     # type: ignore
from scipy.stats import shapiro  # type: ignore
from pathlib import Path
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns   # type: ignore

logger = logging.getLogger(__name__)


def compute_pairwise_differences(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute pairwise differences in MCC between algorithms.
    Args:
        df (pd.DataFrame): DataFrame with columns 'scene_id', 'algorithm', 'mcc'.
    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: Pairwise differences in MCC:
            (hybrid - s2cloudless, hybrid - cloudscoreplus, s2cloudless - cloudscoreplus)
    """
    logger.info("Computing pairwise differences in MCC between algorithms.")

    diff_hy_s2 = df["hybrid"] - df["s2cloudless"]
    diff_hy_cs = df["hybrid"] - df["cloudscoreplus"]
    diff_s2_cs = df["s2cloudless"] - df["cloudscoreplus"]

    return diff_hy_s2, diff_hy_cs, diff_s2_cs

def shapiro_wilk_test(
        pairs: list, 
        diff_pair1: pd.Series, 
        diff_pair2: pd.Series, 
        diff_pair3: pd.Series
        ) -> pd.DataFrame:
    """Test normality of pairwise differences in MCC using Shapiro–Wilk test.
    Args:
        pairs (list): List of algorithm pairs to compare.
    Returns:
        pd.DataFrame: DataFrame with Shapiro-Wilk test results for each pair.
    """
    logger.info("Testing normality of pairwise differences in MCC using Shapiro–Wilk test.")
    if len(pairs) != 3:
        raise ValueError("Exactly three algorithm pairs are required")

    # Shapiro–Wilk normality test for pairwise differences.
    results = []
    for pair, diff in zip(pairs, [diff_pair1, diff_pair2, diff_pair3]):
        logger.info(f"Performing Shapiro–Wilk test for pair: {pair}")
        stat, p = shapiro(diff)
        logger.info(f"Shapiro–Wilk test for {pair}: statistic={stat}, p-value={p}")
        logger.info(f"{pair}: stat = {stat:.4f}, p = {p:.4f}")
        results.append({'algorithm_pair': pair, 'statistic': stat, 'p_value': p})
    
    result_df = pd.DataFrame(results)
    return result_df


def plot_normality(
        diff_pair1: pd.Series, 
        diff_pair2: pd.Series, 
        diff_pair3: pd.Series, 
        output_dir: Path
        ) -> None:
    """Plot histograms and KDEs of pairwise differences in MCC.
    Args:
        diff_pair1 (pd.Series): Series of pairwise differences.
        diff_pair2 (pd.Series): Series of pairwise differences.
        diff_pair3 (pd.Series): Series of pairwise differences.
        output_dir (Path): Directory to save output figures.
    Returns:
        None
    """
    logger.info("Plotting histograms and KDEs of pairwise differences in MCC.")
    diff_series_list = [
        ("hybrid - s2cloudless", diff_pair1),
        ("hybrid - cloudscoreplus", diff_pair2),
        ("s2cloudless - cloudscoreplus", diff_pair3),
    ]
    logger.debug(f"Preparing to plot differences: {[label for label, _ in diff_series_list]}")
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
        # Add Kernel Density Estimate (KDE) to the histogram.
        sns.kdeplot(
            x=diff_series, # Pass the Series directly to x
            color='steelblue', # Assign a color as hue is removed
            linewidth=2
        )

        fig_path = os.path.join(
            output_dir, "hypothesis",
            f"histogram_kde_mcc_diff_{title_label.replace(" ", "_")
            .replace("-", "vs").replace("(", "").replace(")", "")}.png"
            )

        plt.title(f'Distribution of MCC Differences: {title_label}')
        plt.xlabel('MCC Difference')
        plt.ylabel('Density')
        save_figure(plt.gcf(), Path(fig_path))
    logger.info("Histograms and KDEs of pairwise differences in MCC saved.")