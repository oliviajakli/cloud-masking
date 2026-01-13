from src.statistical_tests import run_friedman_test
from pathlib import Path
import pandas as pd
from src.utils.config import load_config

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

def main():
    run_friedman_test(df)
    return "Friedman test completed. Results saved to:", output_dir

if __name__ == "__main__":
    print(main())