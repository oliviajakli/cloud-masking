import logging
import pandas as pd     # type: ignore


logger = logging.getLogger(__name__)

# Check measures of central tendency and dispersion for each algorithm.

def compute_descriptive_stats( df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Core function to compute descriptive statistics for each algorithm.
    Args:
        df (pd.DataFrame): DataFrame containing columns 'algorithm' and specified metrics.
        metrics (list[str]): List of metric names to compute descriptive statistics for.
    Returns:
        pd.DataFrame: Summary DataFrame with median and std dev for each metric per algorithm.
    """
    rows = []
    for algo, group in df.groupby("algorithm"):
        for metric in metrics:
            if metric not in group.columns:
                raise ValueError(f"Metric '{metric}' not found")

            rows.append({
                "algorithm": algo,
                "metric": metric,
                "median": group[metric].median(),
                "std": group[metric].std(),
            })

    return pd.DataFrame(rows)