from src.directional_error import (
    compute_precision_recall_diff,
    summary_table,
    plot_directional_bias
)
from src.utils.config import load_config
from pathlib import Path
import pandas as pd

config = load_config()

input_data = Path(config["paths"]["input"])
random_state = config["statistics"]["random_state"]
output_dir = Path(config["paths"]["output_dir"])


df = pd.read_csv(input_data)

def main():
    # Compute precision-recall difference and add as a new column.
    df = compute_precision_recall_diff(df)
    # Generate summary table and save to CSV.
    summary_table(df)
    # Generate and save plots for directional bias.
    plot_directional_bias(df, output_dir)
    return "Directional analysis completed. Results saved to:", output_dir

if __name__ == "__main__":
    print(main())



