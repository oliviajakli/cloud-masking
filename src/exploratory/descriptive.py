import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config_loader import load_config
from pathlib import Path

config = load_config()

input_data = Path(config["paths"]["input"])
metrics = config["metrics"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

# Check measures of central tendency and dispersion for each algorithm.

# Compute median (what is typical) and standard deviation (consistency) per algorithm.
f1score_summary = df.groupby('algorithm')['f1_score'].agg(['median', 'mean', 'std'])
iou_summary = df.groupby('algorithm')['iou'].agg(['median', 'mean', 'std'])
mcc_summary = df.groupby('algorithm')['mcc'].agg(['median', 'mean', 'std'])
summary_df = pd.concat([f1score_summary, iou_summary, mcc_summary], axis=1, keys=['f1_score', 'iou', 'mcc'])
summary_path = os.path.join(output_dir, 'metrics_summary.csv')
summary_df.to_csv(summary_path)

# Check data distribution by plotting histograms and KDEs for each metric, by algorithm.
for metric in metrics:
    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=df,
        x=metric,
        hue='algorithm',
        bins=15,
        stat='density',
        element='step',
        common_norm=False,
        alpha=0.3
    )

    sns.kdeplot(
        data=df,
        x=metric,
        hue='algorithm',
        common_norm=False,
        linewidth=2
    )

    fig_path = os.path.join(output_dir, f'histogram_kde_{metric}.png')

    plt.title(f'Distribution of {metric.replace("_", " ").title()} (Histogram + KDE)')
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()