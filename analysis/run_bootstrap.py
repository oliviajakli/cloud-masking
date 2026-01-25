from pathlib import Path

from src.bootstrap import (
    compute_tile_mccs_all,
    list_scenes,
    summarize,
    two_level_bootstrap,
)
from src.utils.config import load_config
from src.utils.io import save_csv

config = load_config()

DATA_ROOT = Path(config["paths"]["data_root"])
ALG_FOLDERS = {
    'hybrid': Path(f"{DATA_ROOT}/masks/hybrid"),
    's2cloudless': Path(f"{DATA_ROOT}/masks/s2cloudless"),
    'cloudscoreplus': Path(f"{DATA_ROOT}/masks/cloudscoreplus")
}

TILE_SIZE = 256        # tile size in pixels
B_SCENE = 1000         # per-scene tile bootstrap replicates
B_GLOBAL = 2000        # global paired bootstrap replicates

GT_FOLDER = Path(config["paths"]["reference_masks_dir"])
SEED = config["statistics"]["random_seed"]
OUTPUT_DIR = Path(config["paths"]["output_dir"])

def main() -> tuple[str, Path]:
    """Run two-level bootstrap analysis for cloud detection algorithms.
    Returns:
        tuple[str, Path]: Message and path to output directory.
    """
    # First, identify scenes available across all algorithms.
    scenes_set: set[str] | None = None
    for alg, path in ALG_FOLDERS.items():
        common = set(list_scenes(GT_FOLDER, path))
        scenes_set = common if scenes_set is None else scenes_set.intersection(common)

    if scenes_set is None:
        scenes_set = set()

    scenes = sorted(scenes_set)  # scenes: list[str]
    print(f"Found {len(scenes)} matched scenes.")

    # Next, compute tile MCCs per algorithm per scene.
    scene_tile_mccs = {}
    for alg, folder in ALG_FOLDERS.items():
        scene_tile_mccs[alg] = compute_tile_mccs_all(GT_FOLDER, folder, scenes, tile=TILE_SIZE)

    # Perform two-level bootstrap (per-scene tile bootstrap + global paired bootstrap).
    alg_df, diff_df = two_level_bootstrap(scene_tile_mccs, B_scene=B_SCENE, B_global=B_GLOBAL, seed=SEED)

    # Generate summaries and save results to separate CSV files.
    alg_summary = summarize(alg_df)
    diff_summary = summarize(diff_df)

    save_csv(alg_summary, OUTPUT_DIR / "algorithm_summary.csv", timestamp=False)
    save_csv(diff_summary, OUTPUT_DIR / "pairwise_diff_summary.csv", timestamp=False)
    save_csv(alg_df, OUTPUT_DIR / "alg_bootstrap_raw.csv", timestamp=False)
    save_csv(diff_df, OUTPUT_DIR / "pairwise_diffs_raw.csv", timestamp=False)

    return "Finished. Results saved to:", OUTPUT_DIR

if __name__ == "__main__":
    print(main())