"""
Full statistical analysis pipeline runner.
"""
import logging
from pathlib import Path

import pandas as pd

from pipeline.run_bootstrap import run as run_bootstrap
from pipeline.run_directional import run as run_directional_error
from pipeline.run_evaluation import run as run_evaluation
from pipeline.run_exploratory import run as run_exploratory
from pipeline.run_friedman import run as run_friedman
from pipeline.run_normality_check import run as run_normality_tests
from pipeline.run_posthoc import run as run_posthoc
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_pipeline() -> None:
    setup_logging()
    logger.info("Starting full analysis pipeline.")

    config = load_config()

    paths_cfg = config.get("paths", {})
    validation_cfg = config.get("validation", {})
    stats_cfg = config.get("statistics", {})
    bootstrap_cfg = config.get("bootstrap", {})

    input_path = _resolve_path(paths_cfg["input"])
    output_dir = _resolve_path(paths_cfg["output_dir"])
    metrics = config.get("metrics", validation_cfg.get("metric_columns", []))
    raw_pairs = config["algorithm_pairs"]
    pairs = raw_pairs if isinstance(raw_pairs, list) else [
        (left, right)
        for left, rights in raw_pairs.items()
        for right in rights
    ]
    samples = config.get("samples", [])
    algorithms = config.get("algorithms", list({alg for pair in pairs for alg in pair}))
    masks_dir = _resolve_path(paths_cfg.get("masks_dir", "data/masks"))
    reference_masks_dir = _resolve_path(paths_cfg.get("reference_masks_dir", "data/masks/reference"))
    data_dir = _resolve_path(paths_cfg.get("data", paths_cfg.get("data_root", paths_cfg["output_dir"])))
    random_seed = config.get("random_seed", stats_cfg.get("random_seed", 42))
    tile_size = config.get("tile_size", bootstrap_cfg.get("tile_size", 256))
    b_scene = config.get("b_scene", bootstrap_cfg.get("b_scene", 1000))
    b_global = config.get("b_global", bootstrap_cfg.get("b_global", 2000))
    alg_folders = {
        'hybrid': Path(f"{data_dir}/masks/hybrid"),
        's2cloudless': Path(f"{data_dir}/masks/s2cloudless"),
        'cloudscoreplus': Path(f"{data_dir}/masks/cloudscoreplus")
    }
    

    df = pd.read_csv(input_path)

    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(algorithms)
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
