from src.pairwise_analysis import compute_pairwise_differences, plot_normality, shapiro_wilk_test
from src.utils.config import load_config
from pathlib import Path
import pandas as pd   # type: ignore
import logging

from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

def main(df: pd.DataFrame) -> tuple[str, Path]:
    """Run pairwise analysis including differences and normality tests.
    Args:
        df (pd.DataFrame): DataFrame with algorithm results.
    Returns:
        str: Message indicating where results are saved.
    """
    setup_logging()
    logger.info("Starting pairwise analysis.")
    # Compute pairwise differences.
    diff_hy_s2, diff_hy_cs, diff_s2_cs = compute_pairwise_differences(df, output_dir)
    shapiro_wilk_test(pairs, diff_hy_s2, diff_hy_cs, diff_s2_cs, output_dir)
    logger.info("Generating plots for normality of pairwise differences.")
    plot_normality(diff_hy_s2, diff_hy_cs, diff_s2_cs, output_dir)
    logger.info("Pairwise analysis completed.")
    return "Pairwise analysis completed. Results saved to:", output_dir

if __name__ == "__main__":
    df = pd.read_csv(input_data)
    message, path = main(df)
    print(message, path)