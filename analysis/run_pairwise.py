from src.pairwise_analysis import compute_pairwise_differences, test_normality
from src.utils.config import load_config
from pathlib import Path
import pandas as pd

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

def main():
    compute_pairwise_differences(df, output_dir)
    test_normality(df, pairs, output_dir)
    return "Pairwise analysis completed. Results saved to:", output_dir

if __name__ == "__main__":
    print(main())