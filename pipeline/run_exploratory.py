from pathlib import Path
import logging
import pandas as pd # type: ignore

from src.analysis.exploratory_plots import (compute_descriptive_stats, plot_distributions, 
    plot_boxplots_with_stats,plot_paired_differences, plot_bland_altman, plot_error_maps,
    plot_scatterplot, plot_time_series)
from src.utils.validator import DataValidator
from src.utils.config import load_config
from src.utils.io import save_analysis_results, validate_output_path_for_df
from src.utils.logging import setup_logging   # type: ignore


logger = logging.getLogger(__name__)


def run(
        df: pd.DataFrame, 
        metrics: list[str], 
        pairs: list[tuple[str, str]],
        algorithms: list[str],
        samples: list[str],
        reference_masks_dir: Path,
        config: dict,
        output_dir: Path) -> tuple[str, Path]:
    """Run exploratory analysis and plotting.
    Args:
        df (pd.DataFrame): Input dataframe with evaluation metrics.
        metrics (list[str]): List of metric column names to analyze.
        pairs (list[tuple[str, str]]): List of algorithm pairs for comparison.
        algorithms (list[str]): List of algorithm names.
        samples (list[str]): List of sample identifiers.
        reference_masks_dir (Path): Directory containing reference masks for error maps.
        config (dict): Configuration dictionary with additional parameters.
        output_dir (Path): Directory to save results.
    Returns:
        tuple[str, Path]: Message and path to output directory.
    """
    logger.info("Starting descriptive analysis and plotting...")
    logger.debug(f"Input DataFrame head:\n{df.head()}")
    # Compute cumulative median, mean, and std dev for each algorithm and metric.
    summary_df = compute_descriptive_stats(df, metrics)
    # Validate with actual size before saving.
    validate_output_path_for_df(output_dir, summary_df)
    # Save descriptive statistics summary to CSV with user-friendly error handling.
    save_analysis_results(summary_df.reset_index(), Path(f"{output_dir}/metrics_summary.csv"))
    logger.info("Descriptive statistics summary saved.")
    # Histogram with KDE plots for each metric and algorithm.
    plot_distributions(df, metrics, Path(f"{output_dir}/descriptives"))
    logger.info("Distribution plots created.")
    # Paired boxplots with statistical annotations.
    plot_boxplots_with_stats(df, metrics, pairs, algorithms, Path(f"{output_dir}/boxplots"))
    logger.info("Boxplots with statistical annotations created.")
    # Paired difference plots for clear visualization of mean differences.
    plot_paired_differences(df, metrics, pairs, Path(f"{output_dir}/paired_differences"))
    logger.info("Paired difference plots created.")
    # Bland-Altman plots to assess agreement between algorithm pairs.
    plot_bland_altman(df, pairs, Path(f"{output_dir}/bland_altman"))
    logger.info("Bland-Altman plots created.")
    # Per-pixel error maps for visualizing spatial error distributions.
    plot_error_maps(algorithms, samples, reference_masks_dir, config, Path(f"{output_dir}/error_maps"))
    logger.info("Per-pixel error maps created.")
    # Scatterplots for metric relationships.
    plot_scatterplot(df, metrics, Path(f"{output_dir}/scatterplots"))
    logger.info("Scatterplots created.")
    # Time series plots for metrics over samples.
    plot_time_series(df, metrics, Path(f"{output_dir}/time_series"))
    logger.info("Time series plots created.")
    logger.info("Descriptive analysis and plotting completed.")
    return "Descriptive analysis and exploratory plots completed. Results saved to:", output_dir

def cli() -> None:
    setup_logging()

    try:
        config = load_config()
    except FileNotFoundError as e:
        raise SystemExit(f"Configuration file not found: {e}")
    except Exception as e:
        raise SystemExit(f"Failed to load configuration: {e}")

    try:
        input_data = Path(config["paths"]["metrics_df"])
        metrics = config["metrics"]
        pairs = config["algorithm_pairs"]
        algorithms = config["algorithms"]
        samples = config["samples"]
        reference_masks_dir = Path(config["paths"]["reference_masks_dir"])
        output_dir = Path(config["paths"]["output_root"])
    except KeyError as e:
        raise SystemExit(f"Missing required configuration key: {e}")
    
    df = pd.read_csv(input_data)

    # Light validation to ensure required columns and algorithms are present before plotting.
    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(algorithms)
    )
    validator.validate_light(df, context="exploratory analysis")

    message, path = run(
        df, metrics, pairs, algorithms, samples, 
        reference_masks_dir, config, output_dir)
    print(message, path)


if __name__ == "__main__":
    cli()