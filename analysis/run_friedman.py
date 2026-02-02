from src.friedman import run_friedman_test
from pathlib import Path
import pandas as pd     # type: ignore
import logging
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator

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
    run_friedman_test(df)
    logger.info("Friedman test analysis completed.")
    return "Friedman test completed. Results saved to:", output_dir

if __name__ == "__main__":
    df = pd.read_csv(input_data)
    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(pairs.keys()),
        value_constraints={
            "mcc": lambda s: s.between(-1, 1)
        }
    )
    validator.validate_light(df)
    message, path = main(df)
    print(message, path)