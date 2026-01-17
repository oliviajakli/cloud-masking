from src.descriptive_stats import compute_descriptive_stats
from src.exploratory_plots import plot_distributions, plot_boxplots_with_stats, \
bootstrap_ci, plot_paired_differences, plot_bland_altman, plot_error_maps, \
plot_scatterplot, plot_time_series
from src.utils.config import load_config
from pathlib import Path

import pandas as pd

config = load_config()

input_data = "data/per_scene_evaluation_metrics_20260116_171319.csv"
metrics = config["metrics"]
pairs = config["algorithm_pairs"]
algorithms = config["algorithms"]
samples = config["samples"]
reference_masks = Path(config["paths"]["reference_masks_dir"])
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

def main():
    compute_descriptive_stats(df, output_dir)
    plot_distributions(df, metrics, output_dir)
    plot_boxplots_with_stats(df, metrics, pairs, algorithms, f"{output_dir}/boxplots")
    plot_paired_differences(df, metrics, pairs, f"{output_dir}/paired_differences")
    plot_bland_altman(df, pairs, f"{output_dir}/bland_altman")
    plot_error_maps(algorithms, samples, reference_masks, config, f"{output_dir}/error_maps")
    plot_scatterplot(df, metrics, f"{output_dir}/scatterplots")
    plot_time_series(df, metrics, f"{output_dir}/time_series")
    return "Descriptive analysis and plots completed. Results saved to:", output_dir

if __name__ == "__main__":
    print(main())