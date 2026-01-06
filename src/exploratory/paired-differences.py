import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path

from src.config_loader import load_config

config = load_config()

input_data = Path(config["paths"]["input"])
metrics = config["metrics"]
pairs = config["algorithm_pairs"]
algorithms = config["algorithms"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

def bootstrap_ci(data, n_boot=10000, ci=95, random_state=42):
    rng = np.random.default_rng(random_state)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

for metric in metrics:
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), sharey=True)
    if n_pairs == 1:
        axes = [axes]

    for ax, (a1, a2) in zip(axes, pairs):
        # --- Subset and pivot for this pair ---
        df_sub = df[df["algorithm"].isin([a1, a2])]
        df_pivot = df_sub.pivot(index="scene_id", columns="algorithm", values=metric)

        # Skip if any algorithm missing
        if a1 not in df_pivot.columns or a2 not in df_pivot.columns:
            print(f"Missing data for {a1} or {a2}, skipping.")
            continue

        # --- Compute differences ---
        diffs = df_pivot[a1] - df_pivot[a2]
        mean_diff, ci_low, ci_high = bootstrap_ci(diffs)
        stat, p = wilcoxon(df_pivot[a1], df_pivot[a2])

        # --- Identify outliers (±1.5×IQR) ---
        q1, q3 = np.percentile(diffs, [25, 75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = diffs[(diffs < lower_fence) | (diffs > upper_fence)]

        # --- Plot differences ---
        sns.stripplot(y=diffs, color="gray", size=6, jitter=False, ax=ax)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axhline(mean_diff, color="blue", linestyle="-", linewidth=2)
        ax.fill_between(
            [-0.4, 0.4], ci_low, ci_high,
            color="blue", alpha=0.2, label="95% Bootstrap CI"
        )

        # --- Annotate outliers ---
        for sid, val in outliers.items():
            ax.text(0.05, val, sid, fontsize=9, color="red", va="center")

        # --- Title & labels ---
        ax.set_title(
            f"{a1} − {a2}\nMean Δ = {mean_diff:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]\nWilcoxon p = {p:.4f}",
            fontsize=12
        )
        ax.set_ylabel(f"{metric} Difference ({a1} − {a2})")
        ax.set_xticks([])
        ax.legend(loc="upper right")

    plt.suptitle(f"Paired Differences: {metric}", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_paired_differences.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved paired difference plot for {metric}")