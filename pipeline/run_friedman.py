from src.friedman import run_friedman_test
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator
from src.utils.io import validate_output_path_for_df, save_analysis_results

from pathlib import Path
import pandas as pd     # type: ignore
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
    """Run Friedman test on algorithm results.
    Args:
        df (pd.DataFrame): DataFrame with algorithm results.
    Returns:
        str: Message indicating where results are saved.
    """
    setup_logging()
    logger.info("Starting Friedman test analysis.")
    # Set output path for results.
    output_path = Path(f"{output_dir}/friedman_test_results.csv")
    # Run Friedman test.
    result_df = run_friedman_test(df)
    # Validate output path before saving.
    validate_output_path_for_df(output_path, result_df)
    # Save results with user-friendly error handling.
    save_analysis_results(result_df, output_path)
    logger.info("Friedman test analysis completed.")
    return "Friedman test completed. Results saved to:", output_dir

if __name__ == "__main__":
    df = pd.read_csv(input_data)
    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(pairs.keys())
    )
    validator.validate_light(df, context="Friedman test analysis")
    message, path = main(df)
    print(message, path)