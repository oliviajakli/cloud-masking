import os
import pandas as pd
from src.utils.io import save_csv
from pathlib import Path

# Check measures of central tendency and dispersion for each algorithm.
def compute_descriptive_stats(df, output_dir):
    # Compute median (what is typical) and standard deviation (consistency) per algorithm.
    f1score_summary = df.groupby('algorithm')['f1_score'].agg(['median', 'mean', 'std'])
    iou_summary = df.groupby('algorithm')['iou'].agg(['median', 'mean', 'std'])
    mcc_summary = df.groupby('algorithm')['mcc'].agg(['median', 'mean', 'std'])
    summary_df = pd.concat([f1score_summary, iou_summary, mcc_summary], axis=1, keys=['f1_score', 'iou', 'mcc'])
    # Save summary to CSV.
    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    save_csv(summary_df.reset_index(), Path(summary_path), timestamp=False)