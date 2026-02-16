from src.directional_error import (
    compute_precision_recall_diff,
    summary_table,
    plot_directional_bias
)
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator
from src.utils.io import save_analysis_results, validate_output_path_for_df

from pathlib import Path
import pandas as pd   # type: ignore
import logging


logger = logging.getLogger(__name__)


def main(df: pd.DataFrame, output_dir: Path) -> tuple[str, Path]:
    """Main function to run directional error analysis.
    Args:
        df (pd.DataFrame): Input dataframe with precision and recall data.
        output_dir (Path): Directory to save results.
    Returns:
        tuple[str, Path]: Message and path to output directory.
    """
    setup_logging()
    logger.info("Starting directional error analysis.")
    # Compute precision-recall difference and add as a new column.
    df = compute_precision_recall_diff(df)
    logger.info("Computed precision-recall differences.")
    # Generate summary table and save to CSV.
    summary_df = summary_table(df)
    summary_path = Path(f"{output_dir}/directional_error_summary.csv")
    # Validate output path before saving.
    validate_output_path_for_df(summary_path, summary_df)
    # Save results of summary table as CSV.
    save_analysis_results(summary_df, summary_path)
    logger.info(f"Generated and saved summary table to {summary_path}.")
    # Generate and save plots for directional bias.
    plot_directional_bias(df, output_dir)
    logger.info("Generated directional bias plots.")
    return "Directional analysis completed. Results saved to:", output_dir

def cli():
    # Load and validate configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        raise SystemExit(f"Configuration file not found: {e}")
    except Exception as e:
        raise SystemExit(f"Failed to load configuration: {e}")

    try:
        input_data = Path(config["paths"]["input"])
        output_dir = Path(config["paths"]["output_dir"])
    except KeyError as e:
        raise SystemExit(f"Missing required configuration key: {e}")

    df = pd.read_csv(input_data)
    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(config["algorithm_pairs"].keys())
    )
    validator.validate_light(df, context="directional analysis")
    message, path = main(df, output_dir)
    print(message, path)

if __name__ == "__main__":
    cli()