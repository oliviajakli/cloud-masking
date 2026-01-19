from pathlib import Path

from src.bootstrap import (
    compute_tile_mccs_all,
    list_scenes,
    summarize,
    two_level_bootstrap,
)
from src.utils.config import load_config

config = load_config()

DATA_ROOT = config["paths"]["data_root"]
ALG_FOLDERS = {
    'hybrid': f"{DATA_ROOT}/masks/hybrid",
    's2cloudless': f"{DATA_ROOT}/masks/s2cloudless",
    'cloudscoreplus': f"{DATA_ROOT}/masks/cloudscoreplus"
}

TILE_SIZE = 256        # tile size in pixels
B_SCENE = 1000         # per-scene tile bootstrap replicates
B_GLOBAL = 2000        # global paired bootstrap replicates

GT_FOLDER = config["paths"]["reference_masks_dir"]
SEED = config["statistics"]["random_seed"]
OUTPUT_DIR = Path(config["paths"]["output_dir"])

def main():
    # 1. Identify scenes available across all algorithms
    scenes_set: set[str] | None = None
    for alg, path in ALG_FOLDERS.items():
        common = set(list_scenes(GT_FOLDER, path))
        scenes_set = common if scenes_set is None else scenes_set.intersection(common)

    if scenes_set is None:
        scenes_set = set()

    scenes = sorted(scenes_set)  # scenes: list[str]
    print(f"Found {len(scenes)} matched scenes.")

    # 2. Compute tile MCCs per algorithm per scene
    scene_tile_mccs = {}
    for alg, folder in ALG_FOLDERS.items():
        scene_tile_mccs[alg] = compute_tile_mccs_all(GT_FOLDER, folder, scenes, tile=TILE_SIZE)

    # 3. Two-level bootstrap
    alg_df, diff_df = two_level_bootstrap(scene_tile_mccs, B_scene=B_SCENE, B_global=B_GLOBAL, seed=SEED)

    # 4. Summaries
    alg_summary = summarize(alg_df)
    diff_summary = summarize(diff_df)

    alg_summary.to_csv(f"{OUTPUT_DIR}/algorithm_summary.csv", index=False)
    diff_summary.to_csv(f"{OUTPUT_DIR}/pairwise_diff_summary.csv", index=False)
    alg_df.to_csv(f"{OUTPUT_DIR}/alg_bootstrap_raw.csv", index=False)
    diff_df.to_csv(f"{OUTPUT_DIR}/pairwise_diffs_raw.csv", index=False)

    return "Finished. Results saved to:", OUTPUT_DIR

if __name__ == "__main__":
    print(main())