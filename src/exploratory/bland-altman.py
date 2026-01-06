import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config_loader import load_config
from pathlib import Path

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

for a1, a2 in pairs:
    # Pivot to align scene_id values
    df_pivot = df.pivot(index="scene_id", columns="algorithm", values="cloud_fraction")

    # Compute means and differences
    means = df_pivot[[a1, a2]].mean(axis=1)
    diffs = df_pivot[a1] - df_pivot[a2]

    # Compute statistics
    mean_diff = diffs.mean()
    sd_diff = diffs.std(ddof=1)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    # Create plots.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=means, y=diffs, color="gray", s=70, alpha=0.8, edgecolor="black")
    # Add a regression line to check proportional bias.
    sns.regplot(x=means, y=diffs, scatter=False, color="black", ci=None)

    # Horizontal reference lines
    plt.axhline(mean_diff, color="blue", linestyle="-", linewidth=1.8, label=f"Mean diff = {mean_diff:.3f}")
    plt.axhline(loa_upper, color="red", linestyle="--", linewidth=1.2, label=f"+1.96 SD = {loa_upper:.3f}")
    plt.axhline(loa_lower, color="red", linestyle="--", linewidth=1.2, label=f"-1.96 SD = {loa_lower:.3f}")

    # Shaded band for limits of agreement
    plt.fill_between(means, loa_lower, loa_upper, color="red", alpha=0.05)

    # Labels
    plt.title(f"Bland–Altman Plot: {a1} vs {a2} (Cloud Fraction)", fontsize=14)
    plt.xlabel("Mean Cloud Fraction", fontsize=12)
    plt.ylabel(f"Difference ({a1} - {a2})", fontsize=12)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Text annotations for limits and mean diff
    plt.text(0.05, mean_diff, f"Mean diff = {mean_diff:.3f}", color="blue", fontsize=10)
    plt.text(0.05, loa_upper, f"+1.96 SD = {loa_upper:.3f}", color="red", fontsize=10)
    plt.text(0.05, loa_lower, f"-1.96 SD = {loa_lower:.3f}", color="red", fontsize=10)

    # Save to file
    file_name = f"bland_altman_{a1}_vs_{a2}_cloud_fraction.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {file_name}")