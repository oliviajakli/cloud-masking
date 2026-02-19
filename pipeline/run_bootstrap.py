from pathlib import Path
import logging
from src.analysis.bootstrap import (
    compute_tile_mccs_all,
    list_scenes,
    summarize,
    two_level_bootstrap,
)
from src.utils.config import load_config
from src.utils.io import save_csv
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run(
        reference_masks_dir: Path,
        alg_folders: dict[str, Path],
        random_seed: int,
        tile_size: int,
        b_scene: int,
        b_global: int,
        output_dir: Path
        ) -> tuple[str, Path]:
    """Run two-level bootstrap analysis for cloud detection algorithms.
    
    Returns:
        tuple[str, Path]: Message and path to output directory.
        
    Raises:
        SystemExit: If any critical error occurs during analysis.
    """
    logger.info("Starting two-level bootstrap analysis for cloud detection algorithms.")
    # First, identify scenes available across all algorithms.
    scenes_set: set[str] | None = None
    for alg, path in alg_folders.items():
        try:
            common = set(list_scenes(reference_masks_dir, path))
            scenes_set = common if scenes_set is None else scenes_set.intersection(common)
        except Exception as e:
            logger.error(f"Failed to list scenes for algorithm {alg}: {e}")
            raise

    if scenes_set is None:
        scenes_set = set()

    scenes = sorted(scenes_set)  # scenes: list[str]
    
    # Validate that we have scenes to process
    if not scenes:
        logger.error("No common scenes found across all algorithms. Cannot proceed.")
        raise ValueError("No common scenes found across all algorithms.")
    
    logger.info(f"Found {len(scenes)} matched scenes. Scenes: {scenes}")

    # Next, compute tile MCCs per algorithm per scene.
    scene_tile_mccs = {}
    for alg, folder in alg_folders.items():
        try:
            logger.info(f"Computing tile MCCs for algorithm: {alg}")
            scene_tile_mccs[alg] = compute_tile_mccs_all(
                reference_masks_dir, folder, scenes, tile=tile_size)
        except Exception as e:
            logger.error(f"Failed to compute tile MCCs for algorithm {alg}: {e}")
            raise
    logger.info("Computed tile-level MCCs for all algorithms.")

    # Perform two-level bootstrap (per-scene tile bootstrap + global paired bootstrap).
    try:
        logger.info("Starting two-level bootstrap computation...")
        alg_df, diff_df = two_level_bootstrap(
            scene_tile_mccs, B_scene=b_scene, B_global=b_global, seed=random_seed)
    except Exception as e:
        logger.error(f"Bootstrap computation failed: {e}")
        raise
    logger.info("Completed two-level bootstrap analysis.")

    # Generate summaries and save results to separate CSV files.
    alg_summary = summarize(alg_df)
    diff_summary = summarize(diff_df)
    
    # Validate that summaries are not empty
    if alg_summary.empty or diff_summary.empty:
        logger.error("Summary dataframes are empty. Bootstrap may have failed.")
        raise ValueError("Bootstrap produced empty summary dataframes.")
    
    logger.info("Generated summaries of bootstrapped results.")

    # Save results with proper error handling
    save_csv(alg_summary, Path(f"{output_dir}/bootstrap/algorithm_summary.csv"), timestamp=False)
    save_csv(diff_summary, Path(f"{output_dir}/bootstrap/pairwise_diff_summary.csv"), timestamp=False)
    save_csv(alg_df, Path(f"{output_dir}/bootstrap/alg_bootstrap_raw.csv"), timestamp=False)
    save_csv(diff_df, Path(f"{output_dir}/bootstrap/pairwise_diffs_raw.csv"), timestamp=False)
    message = f"Saved results to {output_dir}."
    logger.info(message)
    return message, output_dir

def cli():
    setup_logging()
    try:
        config = load_config()
    except FileNotFoundError as e:
        raise SystemExit(f"Configuration file not found: {e}")
    except Exception as e:
        raise SystemExit(f"Failed to load configuration: {e}")

    # Validate required config keys
    try:
        data_root = Path(config["paths"]["data"])
        reference_masks_dir = Path(config["paths"]["reference_masks_dir"])
        output_dir = Path(config["paths"]["output_dir"])
        random_seed = config["statistics"]["random_seed"]
        tile_size = config["bootstrap"]["tile_size"]
        b_scene = config["bootstrap"]["b_scene"]
        b_global = config["bootstrap"]["b_global"]
    except KeyError as e:
        raise SystemExit(f"Missing required configuration key: {e}")

    # Validate that required directories exist
    required_dirs = {"data_root": data_root, "reference_masks_dir": reference_masks_dir}
    for dir_name, dir_path in required_dirs.items():
        if not dir_path.exists():
            raise SystemExit(f"Required directory does not exist: {dir_name} -> {dir_path}")

    alg_folders = {
        'hybrid': Path(f"{data_root}/masks/hybrid"),
        's2cloudless': Path(f"{data_root}/masks/s2cloudless"),
        'cloudscoreplus': Path(f"{data_root}/masks/cloudscoreplus")
    }

    # Validate algorithm folders exist
    for alg, path in alg_folders.items():
        if not path.exists():
            raise SystemExit(f"Algorithm folder not found: {alg} -> {path}")

    try:
        message, output_path = run(
            reference_masks_dir, alg_folders, random_seed, 
            tile_size, b_scene, b_global, output_dir
        )

        print(f"{message} {output_path}")
        logger.info("Bootstrap analysis completed successfully.")
    except SystemExit as e:
        logger.error(f"Fatal error: {e}")
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise SystemExit(f"Analysis failed with error: {e}")
    

if __name__ == "__main__":
    cli()