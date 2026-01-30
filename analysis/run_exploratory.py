from src.descriptive_stats import compute_descriptive_stats
from src.exploratory_plots import (plot_distributions, plot_boxplots_with_stats,
    plot_paired_differences, plot_bland_altman, plot_error_maps,plot_scatterplot, 
    plot_time_series)
from src.utils.validator import DataValidator
from src.utils.config import load_config
from pathlib import Path
import logging
import pandas as pd # type: ignore

from src.utils.logging import setup_logging   # type: ignore

logger = logging.getLogger(__name__)

config = load_config()

input_data = Path(config["paths"]["input"])
metrics = config["metrics"]
pairs = config["algorithm_pairs"]
algorithms = config["algorithms"]
samples = config["samples"]
reference_masks = Path(config["paths"]["reference_masks_dir"])
output_dir = Path(config["paths"]["output_dir"])

def main(df):
    setup_logging()
    logger.info("Starting descriptive analysis and plotting...")
    logger.debug(f"Input DataFrame head:\n{df.head()}")
    # Compute cumulative median, mean, and std dev for each algorithm and metric.
    compute_descriptive_stats(df, metrics, output_dir)
    logger.info("Descriptive statistics computed.")
    # Histogram with KDE plots for each metric and algorithm.
    plot_distributions(df, metrics, output_dir)
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
    plot_error_maps(algorithms, samples, reference_masks, config, Path(f"{output_dir}/error_maps"))
    logger.info("Per-pixel error maps created.")
    # Scatterplots for metric relationships.
    plot_scatterplot(df, metrics, Path(f"{output_dir}/scatterplots"))
    logger.info("Scatterplots created.")
    # Time series plots for metrics over samples.
    plot_time_series(df, metrics, Path(f"{output_dir}/time_series"))
    logger.info("Time series plots created.")
    logger.info("Descriptive analysis and plotting completed.")
    return "Descriptive analysis and plots completed. Results saved to:", output_dir

if __name__ == "__main__":
    df = pd.read_csv(input_data)
    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(algorithms),
        value_constraints={
            "precision": lambda s: s.between(0, 1),
            "recall": lambda s: s.between(0, 1),
            "f1_score": lambda s: s.between(0, 1),
            "iou": lambda s: s.between(0, 1),
            "mcc": lambda s: s.between(-1, 1)
        }
    )
    validator.validate_light(df, context="exploratory analysis")
    message, path = main(df)
    print(message, path)