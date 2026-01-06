# Preserve the order of algorithms as they appear in the CSV
# alg_order = df['algorithm'].drop_duplicates().tolist()
# df['algorithm'] = pd.Categorical(df['algorithm'], categories=alg_order, ordered=True)
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from statannotations.Annotator import Annotator

from src.config_loader import load_config

config = load_config()

input_data = Path(config["paths"]["input"])
metrics = config["metrics"]
pairs = config["algorithm_pairs"]
algorithms = config["algorithms"]
output_dir = Path(config["paths"]["output_dir"])


df = pd.read_csv(input_data)

sns.set_theme(style="whitegrid", font_scale=1)

for metric in metrics:
    plt.figure(figsize=(7,5))
    ax = sns.boxplot(data=df, x="algorithm", y=metric, hue="algorithm", palette="pastel", showmeans=True, legend=False)
    sns.stripplot(data=df, x="algorithm", y=metric, color="gray", size=5, alpha=0.6, jitter=True)

    # Compute Wilcoxon signed-rank tests for each pair (paired because same samples)
    p_values = []
    for a1, a2 in pairs:
        # Extract paired values using the same samples
        vals1 = df.loc[df["algorithm"] == a1, metric].values
        vals2 = df.loc[df["algorithm"] == a2, metric].values
        # Run Wilcoxon signed-rank test (paired, non-parametric)
        stat, p = wilcoxon(vals1, vals2)
        p_values.append(p)

    # Add annotations
    annotator = Annotator(ax, pairs, data=df, x="algorithm", y=metric)
    annotator.configure(test=None, text_format="star", loc="inside")
    annotator.set_pvalues(p_values)
    annotator.annotate()

    for i, alg in enumerate(algorithms):
        vals = df.loc[df["algorithm"] == alg, metric]
        median_val = vals.median()
        std_val  = vals.std()
        y_val = float(median_val.iloc[0]) + 0.01 if isinstance(median_val, pd.Series) else median_val + 0.01
        ax.text(i, y_val, f"{median_val:.3f} ± {std_val:.3f}",
                ha="center", fontsize=10, color="black")

    ax.set_title(f"Paired Comparison of {metric} across Algorithms", fontsize=12, pad=15)
    ax.set_ylabel(metric)
    ax.set_xlabel("Algorithm")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{metric}_boxplot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved {metric} plot to: {fig_path}")