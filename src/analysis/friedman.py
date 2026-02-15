from scipy.stats import friedmanchisquare # type: ignore
import pandas as pd     # type: ignore
import logging

logger = logging.getLogger(__name__)

def run_friedman_test(df: pd.DataFrame) -> pd.DataFrame:
    """Run Friedman test on MCC scores across different algorithms.
    Args:
        df (pd.DataFrame): DataFrame containing 'scene_id', 'algorithm', and 'mcc' columns.
    Returns:
        pd.DataFrame: DataFrame with statistic and p-value from the Friedman test.
    """
    logger.info("Running Friedman test on MCC scores across algorithms.")
    # Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
    mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')
    if mcc_wide.shape[1] < 3:
        raise ValueError("Friedman test requires at least three algorithms")
    if mcc_wide.isna().any().any():
        raise ValueError("Missing MCC values detected")

    logger.debug(f"MCC wide format DataFrame for Friedman test:\n{mcc_wide}")
    # Run Friedman test to test whether there is any difference among the algorithms.
    stat, p_friedman = friedmanchisquare(*[mcc_wide[col] for col in mcc_wide.columns])
    logger.info(f"Friedman test: χ² = {stat:.3f}, p-value = {p_friedman:.5f}")
    # Save results to CSV file.
    results_df = pd.DataFrame({
        'statistic': [stat],
        'p_value': [p_friedman]
    })
    logger.info("Friedman test completed and results saved to CSV.")
    return results_df