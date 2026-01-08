import pandas as pd
from src.config_loader import load_config
from scipy.stats import wilcoxon, shapiro
from pathlib import Path

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

# Pivot to wide format (scenes x algorithms). Algorithms will be in alphabetical order.
mcc_wide = df.pivot(index='scene_id', columns='algorithm', values='mcc')

# Compute pairwise differences. Cloud Score+ is at index 0.
# Hybrid method is at index 1. s2cloudless at index 2.
diff_hy_s2 = mcc_wide.iloc[:, 1] - mcc_wide.iloc[:, 2]
diff_hy_cs = mcc_wide.iloc[:, 1] - mcc_wide.iloc[:, 0]
diff_s2_cs = mcc_wide.iloc[:, 2] - mcc_wide.iloc[:, 0]

# Shapiro–Wilk normality test for pairwise differences.
for label, diff in zip(pairs, [diff_hy_s2, diff_hy_cs, diff_s2_cs]):
    stat, p = shapiro(diff)
    print(f"{label}: p = {p:.4f}")