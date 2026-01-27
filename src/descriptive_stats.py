import os
import logging
import pandas as pd     # type: ignore
from src.utils.io import save_csv
from pathlib import Path

logger = logging.getLogger(__name__)

# Check measures of central tendency and dispersion for each algorithm.
def compute_descriptive_stats(df: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """Compute median (what is typical) and standard deviation (consistency) per algorithm.
    Args:
        df (pd.DataFrame): DataFrame containing columns 'algorithm', 'f1_score', 'iou', 'mcc'.
        metrics (list): List of metric names to compute descriptive statistics for.
        output_dir (Path): Directory to save the summary CSV file.
    Returns:
        None
    """
    logger.info("Computing descriptive statistics for each algorithm.")
    # Compute aggregated median and standard deviation for each metric per algorithm.
    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in DataFrame columns.")
        summary = df.groupby('algorithm')[metric].agg(['median', 'std'])
        logger.debug(f"Descriptive statistics for {metric}:\n{summary}")
    summary_df = pd.concat([df.groupby('algorithm')[metric]
                            .agg(['median', 'std']) for metric in metrics], 
                            axis=1, keys=metrics)
    logger.info("Descriptive statistics computed successfully.")
    logger.debug(f"Summary DataFrame:\n{summary_df}")
    # Save summary to CSV.
    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    save_csv(summary_df.reset_index(), Path(summary_path), timestamp=False)