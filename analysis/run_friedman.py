from src.friedman import run_friedman_test
from pathlib import Path
import pandas as pd     # type: ignore
import logging
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

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
    message, path = main(df)
    print(message, path)