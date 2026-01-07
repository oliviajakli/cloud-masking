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

# Ensure numeric and drop missing
df_clean = df.dropna(subset=['cloud_fraction'] + metrics)

# Facet plot: one subplot per metric
for metric in metrics:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df_clean,
        x='cloud_fraction',
        y=metric,
        hue='algorithm',
        alpha=0.7,
        s=60
    )

    # Add trend line
    sns.regplot(
        data=df_clean,
        x='cloud_fraction',
        y=metric,
        scatter=False,
        color='black',
        line_kws={'linestyle': '--'}
    )

    plt.title(f'{metric.replace("_", " ").title()} vs Cloud Fraction')
    plt.xlabel('Cloud Fraction')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cloud_fraction_scatter_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()