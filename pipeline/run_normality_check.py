from src.normality_check import compute_pairwise_differences, plot_normality, shapiro_wilk_test
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator
from src.utils.io import save_analysis_results, save_csv, validate_output_path_for_df

from pathlib import Path
import pandas as pd   # type: ignore
import logging


logger = logging.getLogger(__name__)


def run(df: pd.DataFrame, 
        pairs: list[tuple[str, str]],
        output_dir: Path
        ) -> tuple[str, Path]:
    """Run pairwise analysis including differences and normality tests.
    Args:
        df (pd.DataFrame): DataFrame with algorithm results.
    Returns:
        str: Message indicating where results are saved.
    """
    logger.info("Starting pairwise analysis.")
    # Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
    mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')
    logger.debug(f"MCC wide format DataFrame:\n{mcc_wide}")
    required = {"hybrid", "s2cloudless", "cloudscoreplus"}
    if not required.issubset(mcc_wide.columns):
        raise ValueError(f"Missing algorithms: {required - set(mcc_wide.columns)}")
    # Compute pairwise differences for MCC between algorithm pairs.
    diff_hy_s2, diff_hy_cs, diff_s2_cs = compute_pairwise_differences(mcc_wide)
    # Save differences to output directory.
    pairwise_diff_path = Path(f"{output_dir}/hypothesis/pairwise_mcc_differences.csv")
    diff_df = pd.DataFrame({
        'scene_id': mcc_wide.index,
        'diff_hybrid_s2cloudless': diff_hy_s2,
        'diff_hybrid_cloudscoreplus': diff_hy_cs,
        'diff_s2cloudless_cloudscoreplus': diff_s2_cs
    })
    save_csv(diff_df, pairwise_diff_path)
    logger.info("Pairwise differences in MCC saved.")
    logger.debug(f"Pairwise differences DataFrame:\n{diff_df}")
    # Test normality to determine appropriate statistical tests.
    logger.info("Performing Shapiro-Wilk tests for normality of pairwise differences.")
    result_df = shapiro_wilk_test(pairs, diff_hy_s2, diff_hy_cs, diff_s2_cs, output_dir)
    # Save normality test results as CSV.
    shapiro_wilk_path = Path(f"{output_dir}/hypothesis/shapiro_wilk.csv")
    # Validate output path before saving.
    validate_output_path_for_df(shapiro_wilk_path, result_df)
    # Save results with user-friendly error handling.
    save_analysis_results(result_df, shapiro_wilk_path)
    logger.info("Generating plots for normality of pairwise differences.")
    plot_normality(diff_hy_s2, diff_hy_cs, diff_s2_cs, output_dir)
    logger.info("Pairwise analysis completed.")
    return "Pairwise analysis completed. Results saved to:", output_dir

def cli() -> None:
    setup_logging()
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
    
    df = pd.read_csv(input_data)

    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(config["algorithms"])
    )
    validator.validate_light(df, context="pairwise analysis")
    message, path = run(df, pairs, output_dir)
    print(message, path)

if __name__ == "__main__":
    cli()