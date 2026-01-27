import os
import logging
from pathlib import Path
import itertools
import rasterio     # type: ignore
import pandas as pd   # type: ignore
import numpy as np  # type: ignore
import seaborn as sns   # type: ignore
import matplotlib.pyplot as plt    # type: ignore
import matplotlib.patches as mpatches   # type: ignore
from matplotlib.colors import ListedColormap    # type: ignore
from scipy.stats import wilcoxon    # type: ignore
from statannotations.Annotator import Annotator # type: ignore

from src.utils.plotting import save_figure

logger = logging.getLogger(__name__)

def plot_distributions(df: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """ Check data distribution by plotting histograms and KDEs for each metric, by algorithm.
    Args:
        df (pd.DataFrame): DataFrame containing metrics and algorithm labels.
        metrics (list): List of metric column names to plot.
        output_dir (Path): Directory to save the plots.
    Returns:
        None
    """
    for metric in metrics:
        logger.info(f"Plotting distribution for metric: {metric}")
        sns.set_theme(style="whitegrid", font_scale=1)
        plt.figure(figsize=(7, 5))

        sns.histplot(
            data=df,
            x=metric,
            hue='algorithm',
            bins=15,
            stat='density',
            element='step',
            common_norm=False,
            alpha=0.3
        )
        # Kernel Density Estimate overlay for a comprehensive view of distribution.
        sns.kdeplot(
            data=df,
            x=metric,
            hue='algorithm',
            common_norm=False,
            linewidth=2
        )

        fig_path = os.path.join(output_dir, f'histogram_kde_{metric}.png')

        plt.title(f'Distribution of {metric.replace("_", " ").title()} (Histogram + KDE)')
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel('Density')
        logger.info(f"Saving distribution plot for {metric} to {fig_path}")
        logger.debug(f"Distribution plot data:\n{df[[metric, 'algorithm']].head()}")
        save_figure(plt.gcf(), Path(fig_path))

def plot_boxplots_with_stats(df: pd.DataFrame, metrics: list, pairs: list, algorithms: list, output_dir: Path) -> None:
    """Plot boxplots for each metric across algorithms with statistical annotations.
    Args:
        df (pd.DataFrame): DataFrame containing metrics and algorithm labels.
        metrics (list): List of metric column names to plot.
        pairs (list of tuples): List of algorithm pairs for statistical comparison.
        algorithms (list): List of algorithm names.
        output_dir (Path): Directory to save the plots.
    Returns:
        None
    """
    logger.info("Plotting boxplots with statistical annotations.")
    sns.set_theme(style="whitegrid", font_scale=1)

    for metric in metrics:
        logger.info(f"Plotting boxplot for metric: {metric}")
        plt.figure(figsize=(7,5))
        ax = sns.boxplot(data=df, x="algorithm", y=metric, hue="algorithm", palette="pastel", showmeans=True, legend=False)
        sns.stripplot(data=df, x="algorithm", y=metric, color="gray", size=5, alpha=0.6, jitter=True)

        # Compute Wilcoxon signed-rank tests for each pair (paired because same samples).
        p_values = []
        for a1, a2 in pairs:
            logger.debug(f"Computing Wilcoxon test between {a1} and {a2} for metric {metric}")
            # Extract paired values using the same samples.
            vals1 = df.loc[df["algorithm"] == a1, metric].values
            vals2 = df.loc[df["algorithm"] == a2, metric].values
            logger.debug(f"Values for {a1}: {vals1}")
            logger.debug(f"Values for {a2}: {vals2}")
            # Run Wilcoxon signed-rank test (paired, non-parametric).
            stat, p = wilcoxon(vals1, vals2)
            p_values.append(p)
            logger.info(f"Wilcoxon test between {a1} and {a2} for {metric}: statistic={stat}, p-value={p}")
        logger.debug(f"P-values for {metric}: {p_values}")

        # Add annotations to the boxplot.
        annotator = Annotator(ax, pairs, data=df, x="algorithm", y=metric)
        annotator.configure(test=None, text_format="star", loc="inside")
        annotator.set_pvalues(p_values)
        annotator.annotate()

        for i, alg in enumerate(algorithms):
            logger.debug(f"Annotating median and std for algorithm: {alg}")
            vals = df.loc[df["algorithm"] == alg, metric]
            median_val = vals.median()
            std_val  = vals.std()
            y_val = float(median_val.iloc[0]) + 0.01 if isinstance(median_val, pd.Series) else median_val + 0.01
            ax.text(i, y_val, f"{median_val:.3f} ± {std_val:.3f}",
                    ha="center", fontsize=10, color="black")
            logger.debug(f"{alg} - Median: {median_val}, Std: {std_val}")

        ax.set_title(f"Paired Comparison of {metric} across Algorithms", fontsize=12, pad=15)
        ax.set_ylabel(metric)
        ax.set_xlabel("Algorithm")

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{metric}_boxplot.png")
        save_figure(plt.gcf(), Path(fig_path))
        logger.info(f"Saved boxplot for {metric} to {fig_path}")

def bootstrap_ci(data: np.ndarray, n_boot: int = 10000, ci: float = 95, random_state: int = 42) -> tuple:
    """Compute bootstrap confidence interval for the mean of the data.
    Args:
        data (array-like): Input data for bootstrapping.
        n_boot (int): Number of bootstrap samples.
        ci (float): Confidence interval percentage.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    logger.info("Computing bootstrap confidence interval.")
    rng = np.random.default_rng(random_state)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    logger.info(f"Bootstrap CI computed: Mean={np.mean(data)}, CI=({lower}, {upper})")
    return np.mean(data), lower, upper

def plot_paired_differences(df: pd.DataFrame, metrics: list, pairs: list, output_dir: Path) -> None:
    """Plot paired differences for each metric between algorithm pairs with bootstrap CIs and Wilcoxon p-values.
    Args:
        df (pd.DataFrame): DataFrame containing metrics and algorithm labels.
        metrics (list): List of metric column names to plot.
        pairs (list of tuples): List of algorithm pairs for comparison.
        output_dir (Path): Directory to save the plots.
    Returns:
        None
    """
    logger.info("Plotting paired differences with bootstrap CIs and Wilcoxon p-values.")
    sns.set_theme(style="whitegrid", font_scale=1)

    for metric in metrics:
        logger.info(f"Plotting paired differences for metric: {metric}")
        n_pairs = len(pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), sharey=True)
        if n_pairs == 1:
            axes = [axes]

        for ax, (a1, a2) in zip(axes, pairs):
            logger.debug(f"Processing pair: {a1} vs {a2} for metric {metric}")
            # Subset and pivot for this pair of algorithms.
            df_sub = df[df["algorithm"].isin([a1, a2])]
            df_pivot = df_sub.pivot(index="scene_id", columns="algorithm", values=metric)
            logger.debug(f"Pivoted DataFrame for {a1} vs {a2}:\n{df_pivot.head()}")

            # Skip if any algorithm is missing data.
            if a1 not in df_pivot.columns or a2 not in df_pivot.columns:
                logger.warning(f"Missing data for {a1} or {a2}, skipping.")
                continue

            # Compute differences and statistics.
            logger.debug(f"Computing differences for {a1} - {a2}")
            diffs = df_pivot[a1] - df_pivot[a2]
            mean_diff, ci_low, ci_high = bootstrap_ci(diffs)
            stat, p = wilcoxon(df_pivot[a1], df_pivot[a2])
            logger.info(f"{a1} vs {a2} for {metric}: Mean diff={mean_diff}, 95% CI=({ci_low}, {ci_high}), Wilcoxon p={p}")

            # Identify outliers (±1.5×IQR) for annotation.
            logger.debug("Identifying outliers for annotation.")
            q1, q3 = np.percentile(diffs, [25, 75])
            iqr = q3 - q1
            lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = diffs[(diffs < lower_fence) | (diffs > upper_fence)]
            logger.debug(f"Outliers detected: {outliers}")

            # Plot differences as stripplot.
            logger.info("Plotting differences.")
            sns.stripplot(y=diffs, color="gray", size=6, jitter=False, ax=ax)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(mean_diff, color="blue", linestyle="-", linewidth=2)
            ax.fill_between(
                [-0.4, 0.4], ci_low, ci_high,
                color="blue", alpha=0.2, label="95% Bootstrap CI"
            )

            # Annotate outliers with scene IDs.
            for sid, val in outliers.items():
                ax.text(0.05, val, sid, fontsize=9, color="red", va="center")

            # Title & labels with stats.
            ax.set_title(
                f"{a1} − {a2}\nMean Δ = {mean_diff:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]\nWilcoxon p = {p:.4f}",
                fontsize=12
            )
            ax.set_ylabel(f"{metric} Difference ({a1} − {a2})")
            ax.set_xticks([])
            ax.legend(loc="upper right")

        plt.suptitle(f"Paired Differences: {metric}", fontsize=15, y=1.02)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{metric}_paired_differences.png")
        save_figure(plt.gcf(), Path(fig_path))
        logger.info(f"Saved paired difference plot for {metric}")

def plot_bland_altman(df: pd.DataFrame, pairs: list, output_dir: Path) -> None:
    """Generate Bland-Altman plots for cloud fraction between algorithm pairs.
    Args:
        df (pd.DataFrame): DataFrame containing 'scene_id', 'algorithm', and 'cloud_fraction'.
        pairs (list of tuples): List of algorithm pairs for comparison.
        output_dir (Path): Directory to save the plots.
    Returns:
        None
    """
    logger.info("Plotting Bland-Altman plots for cloud fraction.")
    sns.set_theme(style="whitegrid", font_scale=1)

    for a1, a2 in pairs:
        logger.info(f"Plotting Bland-Altman for {a1} vs {a2}")
        # Pivot to align scene_id values for both algorithms.
        df_pivot = df.pivot(index="scene_id", columns="algorithm", values="cloud_fraction")
        logger.debug(f"Pivoted DataFrame for {a1} vs {a2}:\n{df_pivot.head()}")

        # Compute means and differences for Bland-Altman.
        logger.debug("Computing means and differences for Bland-Altman plot.")
        means = df_pivot[[a1, a2]].mean(axis=1)
        diffs = df_pivot[a1] - df_pivot[a2]

        # Compute statistics for Bland-Altman.
        logger.debug("Computing Bland-Altman statistics.")
        mean_diff = diffs.mean()
        sd_diff = diffs.std(ddof=1)
        loa_upper = mean_diff + 1.96 * sd_diff
        loa_lower = mean_diff - 1.96 * sd_diff
        logger.info(f"Bland-Altman stats for {a1} vs {a2}: Mean diff={mean_diff}, Upper LoA={loa_upper}, Lower LoA={loa_lower}")

        # Create Bland-Altman plot.
        logger.info("Creating Bland-Altman plot.")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=means, y=diffs, color="gray", s=70, alpha=0.8, edgecolor="black")
        # Add a regression line to check proportional bias.
        sns.regplot(x=means, y=diffs, scatter=False, color="black", ci=None)

        # Horizontal reference lines for mean difference and limits of agreement.
        plt.axhline(mean_diff, color="blue", linestyle="-", linewidth=1.8, label=f"Mean diff = {mean_diff:.3f}")
        plt.axhline(loa_upper, color="red", linestyle="--", linewidth=1.2, label=f"+1.96 SD = {loa_upper:.3f}")
        plt.axhline(loa_lower, color="red", linestyle="--", linewidth=1.2, label=f"-1.96 SD = {loa_lower:.3f}")

        # Shaded band for limits of agreement (i.e., confidence interval).
        plt.fill_between(means, loa_lower, loa_upper, color="red", alpha=0.05)

        plt.title(f"Bland–Altman Plot: {a1} vs {a2} (Cloud Fraction)", fontsize=14)
        plt.xlabel("Mean Cloud Fraction", fontsize=12)
        plt.ylabel(f"Difference ({a1} - {a2})", fontsize=12)
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)

        # Text annotations for limits and mean diff on the plot.
        plt.text(0.05, mean_diff, f"Mean diff = {mean_diff:.3f}", color="blue", fontsize=10)
        plt.text(0.05, loa_upper, f"+1.96 SD = {loa_upper:.3f}", color="red", fontsize=10)
        plt.text(0.05, loa_lower, f"-1.96 SD = {loa_lower:.3f}", color="red", fontsize=10)

        file_name = f"bland_altman_{a1}_vs_{a2}_cloud_fraction.png"
        save_figure(plt.gcf(), Path(os.path.join(output_dir, file_name)))
        logger.info(f"Saved: {file_name}")

def plot_error_maps(algorithms: list, samples: list, reference_masks: Path, config: dict, output_dir: Path) -> None:
    """Generate per-pixel error maps for each algorithm and sample compared to reference masks.
    Args:
        algorithms (list): List of algorithm names.
        samples (list): List of sample identifiers.
        reference_mask (Path): Directory containing reference mask files.
        config (dict): Configuration dictionary with paths to algorithm mask directories.
        output_dir (Path): Directory to save the error map plots.
    Returns:
        None
    """
    logger.info("Generating per-pixel error maps.")

    for alg, sample in itertools.product(algorithms, samples):
        logger.info(f"Processing error map for algorithm: {alg}, sample: {sample}")
        reference_path = f"{reference_masks}/{sample}.tif"
        predicted_path = f"{Path(config["paths"][f"{alg}_masks_dir"])}/{sample}.tif"
        out_path = f"{output_dir}/error_map_{alg}_{sample}.tif"

        with rasterio.open(reference_path) as ref_ds, rasterio.open(predicted_path) as pred_ds:
            reference = ref_ds.read(1)
            predicted = pred_ds.read(1)
            logger.debug(f"Reference mask shape: {reference.shape}, Predicted mask shape: {predicted.shape}")

        # Initialize error map and classify pixels into TP, TN, FP, FN.
        logger.debug("Classifying pixels into TP, TN, FP, FN.")
        error_map = np.zeros_like(reference, dtype=np.uint8)
        error_map[(reference == 1) & (predicted == 1)] = 1
        error_map[(reference == 0) & (predicted == 0)] = 2
        error_map[(reference == 0) & (predicted == 1)] = 3
        error_map[(reference == 1) & (predicted == 0)] = 4
        logger.debug(f"Error map unique values: {np.unique(error_map)}")

        colors = ["black", "lime", "gray", "red", "orange"]  # BG, TP, TN, FP, FN
        labels = ["Background", "True Positive", "True Negative", "False Positive", "False Negative"]
        cmap = ListedColormap(colors[1:])

        logger.info("Plotting error map.")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(error_map, cmap=cmap, interpolation="none")
        ax.set_title(f"Error Map – {alg.upper()} ({sample})", fontsize=14)
        ax.axis("off")

        patches = [mpatches.Patch(color=colors[i+1], label=labels[i+1]) for i in range(4)]
        ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
        save_figure(plt.gcf(), Path(out_path))
        logger.info(f"Per-pixel error map for {alg}, {sample} saved in {output_dir} folder.")

def plot_scatterplot(df: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """Plot scatterplots of metrics against cloud fraction, colored by algorithm.
    Args:
        df (pd.DataFrame): DataFrame containing 'cloud_fraction', metrics, and 'algorithm
        metrics (list): List of metric column names to plot.
        output_dir (Path): Directory to save the plots.
    Returns:
        None
    """
    logger.info("Plotting scatterplots of metrics vs cloud fraction.")
    sns.set_theme(style="whitegrid", font_scale=1)
    df_clean = df.dropna(subset=['cloud_fraction'] + metrics)
    logger.debug(f"DataFrame after dropping NaNs:\n{df_clean.head()}")

    # Facet plot: one subplot per metric.
    for metric in metrics:
        logger.info(f"Plotting scatterplot for metric: {metric}")
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df_clean,
            x='cloud_fraction',
            y=metric,
            hue='algorithm',
            alpha=0.7,
            s=60
        )

        # Add trend line using regression.
        sns.regplot(
            data=df_clean,
            x='cloud_fraction',
            y=metric,
            scatter=False,
            color='black',
            line_kws={'linestyle': '--'}
        )

        plt.title(f'{metric.replace("_", " ").title()} vs Cloud Fraction')
        plt.xlabel('Cloud Fraction')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend(title='Algorithm')
        fig_path = os.path.join(output_dir, f'cloud_fraction_scatter_{metric}.png')
        save_figure(plt.gcf(), Path(fig_path))
        logger.info(f"Scatterplot for {metric} saved in {output_dir} folder.")

def plot_time_series(df: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """Plot time series of metrics over time, colored by algorithm.
    Args:
        df (pd.DataFrame): DataFrame containing 'scene_id', metrics, and 'algorithm
        metrics (list): List of metric column names to plot.
        output_dir (Path): Directory to save the plots.
    Returns:
        None
    """
    logger.info("Plotting time series of metrics over time.")
    sns.set_theme(style="whitegrid", font_scale=1)
    # Replace 'date' with the actual column name if different (e.g., 'acquisition_date')
    df["date"] = df["scene_id"].astype(str).str.extract(r"(\d{6})")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    logger.debug(f"DataFrame with date column:\n{df[['scene_id', 'date']].head()}")

    # Long-form transformation for easier plotting.
    melted = df.melt(
        id_vars=["scene_id", "algorithm", "date"],
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    )
    logger.debug(f"Melted DataFrame for time series:\n{melted.head()}")

    for metric in metrics:
        logger.info(f"Plotting time series for metric: {metric}")
        subset = melted[melted["metric"] == metric]
        logger.debug(f"Subset DataFrame for {metric}:\n{subset.head()}")

        logger.info(f"Creating time series plot for {metric}.")
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=subset,
            x="date",
            y="value",
            hue="algorithm",
            errorbar="sd",        # show ±1 standard deviation as confidence interval.
            marker="o",
            linewidth=1.8,
            alpha=0.9
        )

        plt.title(f"{metric.upper()} Over Time by Algorithm", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"time_series_{metric}_by_algorithm.png")
        save_figure(plt.gcf(), Path(out_path))
        logger.info(f"Time series plot for {metric} saved in {output_dir} folder.")