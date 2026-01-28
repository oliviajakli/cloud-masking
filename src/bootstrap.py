import os
from pathlib import Path
import numpy as np  # type: ignore
import pandas as pd     # type: ignore
import rasterio  # type: ignore
from glob import glob
from sklearn.metrics import matthews_corrcoef   # type: ignore
from tqdm import tqdm   # type: ignore
import logging

logger = logging.getLogger(__name__)

# Tile-based block bootstrap.
# Divide each sample into spatial tiles (256×256 px) and resample tiles within
# a sample. Bootstrap the 15 samples with replacement, recompute the metric
# median for each algorithm, and compute bootstrap percentiles for metrics and
# paired differences. Two-level bootstrap with 1,000 internal repetitions
# (tile-level), and 2,000 repetitions globally.

def list_scenes(gt_folder: Path, alg_folder: Path) -> list[str]:
    """
    Retrieve sorted list of scenes from the given folders.
    Args:
        gt_folder (Path): Path to ground truth folder.
        alg_folder (Path): Path to algorithm results folder.
    Returns:
        list[str]: Sorted list of common scene filenames.
    """
    logger.debug(f"Listing scenes in {gt_folder} and {alg_folder}.")
    gt_files = {os.path.basename(p): p for p in glob(os.path.join(gt_folder, "*.tif"))}
    alg_files = {os.path.basename(p): p for p in glob(os.path.join(alg_folder, "*.tif"))}
    common = sorted(set(gt_files.keys()).intersection(alg_files.keys()))
    logger.debug(f"Found common scenes: {common}")
    return common

def read_raster(path: Path) -> np.ndarray:
    """Read a raster file and return its array.
    Args:
        path (Path): Path to the raster file.
    Returns:
        np.ndarray: Array representation of the raster.
    """
    logger.debug(f"Reading raster file: {path}")
    with rasterio.open(path) as src:
        arr = src.read(1)
    logger.debug(f"Raster read successfully: {path}")
    return arr

def tile_array(arr: np.ndarray, tile_h: int, tile_w: int) -> list[np.ndarray]:
    """Divide a 2D array into non-overlapping tiles.
    Args:
        arr (np.ndarray): 2D input array.
        tile_h (int): Height of each tile.
        tile_w (int): Width of each tile.
    Returns:
        list[np.ndarray]: List of 2D tiles.
    """
    logger.debug(f"Tiling array of shape {arr.shape} into tiles of size ({tile_h}, {tile_w}).")
    H, W = arr.shape
    tiles = []
    for i in range(0, H, tile_h):
        for j in range(0, W, tile_w):
            tile = arr[i:i+tile_h, j:j+tile_w]
            if tile.shape == (tile_h, tile_w):
                tiles.append(tile)
    logger.debug(f"Total tiles created: {len(tiles)}")
    return tiles

def safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Matthews correlation coefficient safely, i.e. prevent crashes.
    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.
    Returns:
        float: Matthews correlation coefficient or NaN if undefined.
    """
    logger.debug("Computing safe MCC.")
    y_true = (y_true > 0).astype(np.int8).ravel()
    y_pred = (y_pred > 0).astype(np.int8).ravel()
    logger.debug(f"y_true unique values: {np.unique(y_true)}, y_pred unique values: {np.unique(y_pred)}")
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    logger.info("Calculating MCC.")
    return matthews_corrcoef(y_true, y_pred)

def mcc_per_tile(gt_arr: np.ndarray, pred_arr: np.ndarray, tile: int = 256) -> np.ndarray:
    """Compute MCC for each tile in the given arrays.
    Args:
        gt_arr (np.ndarray): Ground truth array.
        pred_arr (np.ndarray): Predicted array.
        tile (int): Tile size (assumed square).
    Returns:
        np.ndarray: Array of MCC values per tile.
    """
    logger.debug("Computing MCC per tile.")
    gt_tiles = tile_array(gt_arr, tile, tile)
    pred_tiles = tile_array(pred_arr, tile, tile)
    mccs = []
    for a, b in zip(gt_tiles, pred_tiles):
        mccs.append(safe_mcc(a, b))
    logger.debug(f"MCC per tile computed: {mccs}")
    return np.array(mccs)

def bootstrap_scene_tiles(mcc_tiles: np.ndarray, B: int = 1000) -> np.ndarray:
    """Bootstrap median MCC from tile-level MCCs for a single scene.
    Args:
        mcc_tiles (np.ndarray): Array of MCC values per tile for a scene.
        B (int): Number of bootstrap samples.
    Returns:
        np.ndarray: Array of bootstrapped median MCC values.
    """
    logger.info("Bootstrapping median MCC from tile-level MCCs.")
    vals = mcc_tiles[~np.isnan(mcc_tiles)]
    if vals.size == 0:
        return np.full(B, np.nan)
    n = len(vals)
    boots = []
    for _ in range(B):
        sample = np.random.choice(vals, size=n, replace=True)
        boots.append(np.median(sample))
    logger.debug(f"Bootstrapped median MCCs: {boots}")
    return np.array(boots)

def compute_tile_mccs_all(gt_folder: Path, alg_folder: Path, scenes: list[str], tile: int = 256) -> dict[str, dict[str, np.ndarray]]:
    """Compute tile-level MCCs for all algorithms and scenes.
    Args:
        gt_folder (Path): Path to ground truth folder.
        alg_folder (Path): Path to algorithm results folder.
        scenes (list[str]): List of scene filenames.
        tile (int): Tile size (assumed square).
    Returns:
        dict[str, dict[str, np.ndarray]]: Nested dictionary of MCC arrays per scene per algorithm.
    """
    logger.info(f"Computing tile-level MCCs for all scenes in {alg_folder}.")
    result = {}
    for s in tqdm(scenes, desc=f"Processing {os.path.basename(alg_folder)}"):
        gt = read_raster(Path(os.path.join(gt_folder, s)))
        pred = read_raster(Path(os.path.join(alg_folder, s)))
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch in {s}")
        result[s] = mcc_per_tile(gt, pred, tile=tile)
    logger.info(f"Completed computing tile-level MCCs for {alg_folder}.")
    return result

def two_level_bootstrap(scene_tile_mccs: dict[str, dict[str, np.ndarray]], B_scene: int = 1000, B_global: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform two-level bootstrap to compute global metrics and paired differences.
    Args:
        scene_tile_mccs (dict[str, dict[str, np.ndarray]]): Nested dictionary of MCC arrays per scene per algorithm.
        B_scene (int): Number of bootstrap samples at the scene level.
        B_global (int): Number of global bootstrap samples.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames of bootstrapped global metrics and paired differences.
    """
    logger.info("Starting two-level bootstrap analysis.")
    np.random.seed(seed)
    algs = list(scene_tile_mccs.keys())
    scenes = sorted(scene_tile_mccs[algs[0]].keys())
    logger.debug(f"Algorithms: {algs}, Scenes: {scenes}")
    S = len(scenes)

    # Per-scene bootstrap for each algorithm (inner layer).
    scene_boots: dict[str, dict[str, np.ndarray]] = {alg: {} for alg in algs}
    for alg in algs:
        for s in scenes:
            scene_boots[alg][s] = bootstrap_scene_tiles(scene_tile_mccs[alg][s], B=B_scene)

    # Global paired bootstrap (outer layer, drives inferential results).
    alg_global = {alg: np.zeros(B_global) for alg in algs}
    for b in tqdm(range(B_global), desc="Global bootstrap"):
        idxs = np.random.randint(0, S, size=S)
        for alg in algs:
            vals = [scene_boots[alg][scenes[i]][np.random.randint(0, B_scene)] for i in idxs]
            alg_global[alg][b] = np.nanmedian(vals)
    logger.info("Completed two-level bootstrap analysis.")

    # Paired differences between algorithms.
    diffs = {}
    for i in range(len(algs)):
        for j in range(i+1, len(algs)):
            a, bname = algs[i], algs[j]
            diffs[f"{a}_minus_{bname}"] = alg_global[a] - alg_global[bname]
    logger.info("Computed paired differences between algorithms.")
    return pd.DataFrame(alg_global), pd.DataFrame(diffs)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize bootstrapped results with median and 95% CI.
    Args:
        df (pd.DataFrame): DataFrame with bootstrapped results.
    Returns:
        pd.DataFrame: Summary DataFrame with median and confidence intervals.
    """
    logger.info("Summarizing bootstrapped results.")
    rows = []
    for col in df.columns:
        med = np.nanmedian(df[col])
        low, high = np.nanpercentile(df[col], [2.5, 97.5])
        rows.append({"metric": col, "median": med, "ci_lower": low, "ci_upper": high})
    logger.debug(f"Summary rows: {rows}")
    return pd.DataFrame(rows)