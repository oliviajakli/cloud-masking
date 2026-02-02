from src.normality_check import compute_pairwise_differences, plot_normality, shapiro_wilk_test
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator

from pathlib import Path
import pandas as pd   # type: ignore
import logging


logger = logging.getLogger(__name__)

# Load and validate configuration
try:
    config = load_config()
except FileNotFoundError as e:
    raise SystemExit(f"Configuration file not found: {e}")
except Exception as e:
    raise SystemExit(f"Failed to load configuration: {e}")

try:
    input_data = Path(config["paths"]["input"])
    pairs = config["algorithm_pairs"]
    output_dir = Path(config["paths"]["output_dir"])
except KeyError as e:
    raise SystemExit(f"Missing required configuration key: {e}")

def main(df: pd.DataFrame) -> tuple[str, Path]:
    """Run pairwise analysis including differences and normality tests.
    Args:
        df (pd.DataFrame): DataFrame with algorithm results.
    Returns:
        str: Message indicating where results are saved.
    """
    setup_logging()
    logger.info("Starting pairwise analysis.")
    # Compute pairwise differences for MCC between algorithm pairs.
    diff_hy_s2, diff_hy_cs, diff_s2_cs = compute_pairwise_differences(df, output_dir)
    # Test normality to determine appropriate statistical tests.
    logger.info("Performing Shapiro-Wilk tests for normality of pairwise differences.")
    shapiro_wilk_test(pairs, diff_hy_s2, diff_hy_cs, diff_s2_cs, output_dir)
    logger.info("Generating plots for normality of pairwise differences.")
    plot_normality(diff_hy_s2, diff_hy_cs, diff_s2_cs, output_dir)
    logger.info("Pairwise analysis completed.")
    return "Pairwise analysis completed. Results saved to:", output_dir

if __name__ == "__main__":
    df = pd.read_csv(input_data)
    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(config["algorithms"]),
        value_constraints={
            "mcc": lambda s: s.between(-1, 1)
        }
    )
    validator.validate_light(df, context="pairwise analysis")
    message, path = main(df)
    print(message, path)