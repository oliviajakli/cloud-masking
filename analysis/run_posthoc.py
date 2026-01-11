from src.posthoc_tests import run_posthoc_wilcoxon
from src.utils.config import load_config
from pathlib import Path
import pandas as pd

config = load_config()

input_data = Path(config["paths"]["input"])
pairs = config["algorithm_pairs"]
output_dir = Path(config["paths"]["output_dir"])

df = pd.read_csv(input_data)

run_posthoc_wilcoxon(df, pairs)