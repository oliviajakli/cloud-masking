import numpy as np  # type: ignore
import pandas as pd     # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns   # type: ignore
import os
from pathlib import Path
from scipy.stats import wilcoxon    # type: ignore
from src.utils.plotting import save_figure
from src.utils.io import save_csv

def compute_precision_recall_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column: pr_diff = precision - recall.
    Args:
        df (pd.DataFrame): DataFrame with 'precision' and 'recall' columns.
    Returns:
        pd.DataFrame: DataFrame with added 'pr_diff' column.
    """
    df = df.copy()
    df["pr_diff"] = df["precision"] - df["recall"]
    return df

def bootstrap_median_ci(x: np.ndarray, n_boot: int = 5000, ci: int = 95, random_state: int = 42) -> tuple[float, float, float]:
    """Computes bootstrap confidence interval for the median of x.
    Args:
        x (np.ndarray): Input data array.
        n_boot (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple[float, float, float]: Median, lower CI bound, upper CI bound.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    boot_medians = [
        np.median(rng.choice(x, size=len(x), replace=True))
        for _ in range(n_boot)
    ]

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_medians, alpha)
    upper = np.percentile(boot_medians, 100 - alpha)

    return np.median(x), lower, upper

def wilcoxon_vs_zero(x: np.ndarray) -> float:
    """Performs Wilcoxon signed-rank test against zero.
    Args:
        x (np.ndarray): Input data array.
    Returns:
        float: p-value from the Wilcoxon test.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    if np.allclose(x, 0):
        return np.nan

    stat, p = wilcoxon(x, zero_method="wilcox", alternative="two-sided")
    return p

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Computes summary statistics for each algorithm in the dataframe.
    Args:
        df (pd.DataFrame): Input dataframe with 'pr_diff' column.
    Returns:
        pd.DataFrame: Summary statistics dataframe.
    """
    summary_rows = []

    for algo, g in df.groupby("algorithm"):
        median, ci_lo, ci_hi = bootstrap_median_ci(g["pr_diff"])
        p_value = wilcoxon_vs_zero(g["pr_diff"])

        summary_rows.append({
            "algorithm": algo,
            "median_pr_diff": median,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "wilcoxon_p": p_value,
            "pct_commission": np.mean(g["pr_diff"] > 0) * 100
        })

    summary_df = pd.DataFrame(summary_rows)
    # Save summary table
    summary_path = os.path.join("results", "directional_error_summary.csv")
    save_csv(summary_df, Path(summary_path))
    return summary_df

def plot_directional_bias(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates and saves a violin plot of precision-recall differences.
    Args:
        df (pd.DataFrame): Input dataframe with 'pr_diff' column.
        output_dir (Path): Directory to save the plot.
    Returns:
        None
    """
    plt.figure(figsize=(9, 5))

    sns.violinplot(
        data=df,
        x="algorithm",
        y="pr_diff",
        inner=None,
        cut=0
    )

    sns.stripplot(
        data=df,
        x="algorithm",
        y="pr_diff",
        color="black",
        alpha=0.6,
        jitter=True
    )

    plt.axhline(0, color="red", linestyle="--", linewidth=1)

    plt.ylabel("Precision − Recall")
    plt.title("Commission vs. Omission Bias per Algorithm")

    fig_path = os.path.join(output_dir, "directional_bias_violinplot.png")
    save_figure(plt.gcf(), Path(fig_path))