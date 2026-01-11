from src.pairwise_analysis import compute_pairwise_differences, test_normality
from src.utils.config import load_config
from pathlib import Path
import pandas as pd

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

compute_pairwise_differences(df)

test_normality()
