import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.config_loader import load_config
from pathlib import Path

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
metrics = config["metrics"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

# Replace 'date' with the actual column name if different (e.g., 'acquisition_date')
df["date"] = df["scene_id"].astype(str).str.extract(r"(\d{6})")
df["date"] = pd.to_datetime(df["date"], format="%Y%m")


# Long-form transformation for easier plotting.
melted = df.melt(
    id_vars=["scene_id", "algorithm", "date"],
    value_vars=metrics,
    var_name="metric",
    value_name="value"
)

# Loop through each metric to create time series plots.
for metric in metrics:
    subset = melted[melted["metric"] == metric]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=subset,
        x="date",
        y="value",
        hue="algorithm",
        errorbar="sd",            # show ±1 standard deviation as confidence interval
        marker="o",
        linewidth=1.8,
        alpha=0.9
    )

    # Add title and labels
    plt.title(f"{metric.upper()} Over Time by Algorithm", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()

    # Save each metric plot
    out_path = os.path.join(output_dir, f"time_series_{metric}_by_algorithm.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()