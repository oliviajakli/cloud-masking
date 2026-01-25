from scipy.stats import friedmanchisquare
from pathlib import Path
import pandas as pd
from src.utils.io import save_csv

def run_friedman_test(df: pd.DataFrame) -> tuple:
    """Run Friedman test on MCC scores across different algorithms.
    Args:
        df (pd.DataFrame): DataFrame containing 'scene_id', 'algorithm', and 'mcc' columns.
    Returns:
        tuple: statistic and p-value from the Friedman test.
    """
    # Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
    mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')
    # Run Friedman test to test whether there is any difference among the algorithms.
    stat, p_friedman = friedmanchisquare(*[mcc_wide[col] for col in mcc_wide.columns])
    print(f"Friedman test: χ² = {stat:.3f}, p-value = {p_friedman:.5f}")
    # Save results to CSV file.
    results_df = pd.DataFrame({
        'statistic': [stat],
        'p_value': [p_friedman]
    })
    save_csv(results_df, path=Path("results/friedman_test_results.csv"))
    return stat, p_friedman