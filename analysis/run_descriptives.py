from src.descriptive_stats import compute_descriptive_stats
from src.exploratory_plots import plot_distributions, plot_boxplots_with_stats, \
bootstrap_ci, plot_paired_differences, plot_bland_altman, plot_error_maps, \
plot_scatterplot, plot_time_series
from src.utils.config import load_config
from pathlib import Path

import pandas as pd

config = load_config()

input_data = Path(config["paths"]["input"])
metrics = config["metrics"]
pairs = config["algorithm_pairs"]
algorithms = config["algorithms"]
samples = config["samples"]
reference_masks = Path(config["paths"]["reference_dir"])
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

compute_descriptive_stats(df, output_dir)
plot_distributions(df, metrics, output_dir)
plot_boxplots_with_stats(df, metrics, pairs, algorithms, output_dir)
bootstrap_ci(df, random_state=config["statistics"]["random_state"])
plot_paired_differences(df, metrics, pairs, output_dir)
plot_bland_altman(df, pairs, output_dir)
plot_error_maps(algorithms, samples, reference_masks, config, output_dir)
plot_scatterplot(df, metrics, output_dir)
plot_time_series(df, metrics, output_dir)