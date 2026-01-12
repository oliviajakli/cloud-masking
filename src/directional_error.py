import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from scipy.stats import wilcoxon
from utils.plotting import save_figure
from utils.io import save_csv

def compute_precision_recall_diff(df):
    """
    Adds a column:
        pr_diff = precision - recall
    """
    df = df.copy()
    df["pr_diff"] = df["precision"] - df["recall"]
    return df

def bootstrap_median_ci(x, n_boot=5000, ci=95, random_state=42):
    """Computes bootstrap confidence interval for the median of x."""
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

def wilcoxon_vs_zero(x):
    """Performs Wilcoxon signed-rank test against zero."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    if np.allclose(x, 0):
        return np.nan

    stat, p = wilcoxon(x, zero_method="wilcox", alternative="two-sided")
    return p

def summary_table(df):
    """Computes summary statistics for each algorithm in the dataframe."""
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

def plot_directional_bias(df, output_dir):
    """Generates and saves a violin plot of precision-recall differences."""
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
    save_figure(plt, Path(fig_path))