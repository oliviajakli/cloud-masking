"""
Full statistical analysis pipeline runner.
"""
from pathlib import Path
import logging
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator

from pipeline.run_evaluation import run as run_evaluation
from pipeline.run_exploratory import run as run_exploratory
from pipeline.run_normality_check import run as run_normality_tests
from pipeline.run_friedman import run as run_friedman
from pipeline.run_posthoc import run as run_posthoc
from pipeline.run_bootstrap import run as run_bootstrap
from pipeline.run_directional_error import run as run_directional_error

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    setup_logging()
    logger.info("Starting full analysis pipeline.")

    config = load_config()

    input_path = Path(config["paths"]["input"])
    output_dir = Path(config["paths"]["output_dir"])
    metrics = config["metrics"]
    pairs = config["algorithm_pairs"]
    samples = config["samples"]
    algorithms = config["paths"]["algorithms"]
    masks_dir = Path(config["paths"]["masks_dir"])
    reference_masks_dir = Path(config["paths"]["reference_masks_dir"])
    data_dir = Path(config["paths"]["data"])
    random_seed = config["random_seed"]
    tile_size = config["tile_size"]
    b_scene = config["b_scene"]
    b_global = config["b_global"]
    alg_folders = {
        'hybrid': Path(f"{data_dir}/masks/hybrid"),
        's2cloudless': Path(f"{data_dir}/masks/s2cloudless"),
        'cloudscoreplus': Path(f"{data_dir}/masks/cloudscoreplus")
    }
    

    df = pd.read_csv(input_path)

    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(pairs.keys())
    )

    validator.validate_light(df, context="Full pipeline")

    # Run all analyses
    run_evaluation(algorithms, masks_dir, reference_masks_dir, data_dir, validator)
    run_exploratory(df, metrics, pairs, algorithms, samples, reference_masks_dir, config, output_dir)
    run_normality_tests(df, pairs, output_dir)
    run_friedman(df, output_dir)
    run_posthoc(df, pairs, random_seed, output_dir)
    run_bootstrap(reference_masks_dir, alg_folders, random_seed, tile_size, b_scene, b_global, output_dir)
    run_directional_error(df, output_dir)

    logger.info("Full analysis pipeline completed.")


if __name__ == "__main__":
    run_pipeline()
