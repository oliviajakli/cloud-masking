from src.directional_error import (
    compute_precision_recall_diff,
    summary_table,
    plot_directional_bias
)
from src.utils.config import load_config
from src.utils.logging import setup_logging
from pathlib import Path
import pandas as pd   # type: ignore
import logging

logger = logging.getLogger(__name__)

config = load_config()

input_data = Path(config["paths"]["input"])
output_dir = Path(config["paths"]["output_dir"])


def main(df: pd.DataFrame) -> tuple[str, Path]:
    """Main function to run directional error analysis.
    Args:
        df (pd.DataFrame): Input dataframe with precision and recall data.
    Returns:
        tuple[str, Path]: Message and path to output directory.
    """
    setup_logging()
    logger.info("Starting directional error analysis.")
    # Compute precision-recall difference and add as a new column.
    df = compute_precision_recall_diff(df)
    logger.info("Computed precision-recall differences.")
    # Generate summary table and save to CSV.
    summary_table(df)
    logger.info("Generated summary table.")
    # Generate and save plots for directional bias.
    plot_directional_bias(df, output_dir)
    logger.info("Generated directional bias plots.")
    return "Directional analysis completed. Results saved to:", output_dir

if __name__ == "__main__":
    df = pd.read_csv(input_data)
    message, path = main(df)
    print(message, path)



