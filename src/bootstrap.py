import os
import numpy as np
import pandas as pd
import rasterio
from glob import glob
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

# Tile-based block bootstrap.
# Divide each sample into spatial tiles (256×256 px) and resample tiles within
# a sample. Bootstrap the 15 samples with replacement, recompute the metric
# median for each algorithm, and compute bootstrap percentiles for metrics and
# paired differences. Two-level bootstrap with 1,000 internal repetitions
# (tile-level), and 2,000 repetitions globally.

def list_scenes(gt_folder, alg_folder):
    """
    Retrieve sorted list of scenes from the given folders.
    """
    gt_files = {os.path.basename(p): p for p in glob(os.path.join(gt_folder, "*.tif"))}
    alg_files = {os.path.basename(p): p for p in glob(os.path.join(alg_folder, "*.tif"))}
    common = sorted(set(gt_files.keys()).intersection(alg_files.keys()))
    return common

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
    return arr

def tile_array(arr, tile_h, tile_w):
    H, W = arr.shape
    tiles = []
    for i in range(0, H, tile_h):
        for j in range(0, W, tile_w):
            tile = arr[i:i+tile_h, j:j+tile_w]
            if tile.shape == (tile_h, tile_w):
                tiles.append(tile)
    return tiles

def safe_mcc(y_true, y_pred):
    y_true = (y_true > 0).astype(np.int8).ravel()
    y_pred = (y_pred > 0).astype(np.int8).ravel()
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    return matthews_corrcoef(y_true, y_pred)

def mcc_per_tile(gt_arr, pred_arr, tile=256):
    gt_tiles = tile_array(gt_arr, tile, tile)
    pred_tiles = tile_array(pred_arr, tile, tile)
    mccs = []
    for a, b in zip(gt_tiles, pred_tiles):
        mccs.append(safe_mcc(a, b))
    return np.array(mccs)

def bootstrap_scene_tiles(mcc_tiles, B=1000):
    vals = mcc_tiles[~np.isnan(mcc_tiles)]
    if vals.size == 0:
        return np.full(B, np.nan)
    n = len(vals)
    boots = []
    for _ in range(B):
        sample = np.random.choice(vals, size=n, replace=True)
        boots.append(np.median(sample))
    return np.array(boots)

def compute_tile_mccs_all(gt_folder, alg_folder, scenes, tile=256):
    result = {}
    for s in tqdm(scenes, desc=f"Processing {os.path.basename(alg_folder)}"):
        gt = read_raster(os.path.join(gt_folder, s))
        pred = read_raster(os.path.join(alg_folder, s))
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch in {s}")
        result[s] = mcc_per_tile(gt, pred, tile=tile)
    return result

def two_level_bootstrap(scene_tile_mccs, B_scene=1000, B_global=2000, seed=0):
    np.random.seed(seed)
    algs = list(scene_tile_mccs.keys())
    scenes = sorted(scene_tile_mccs[algs[0]].keys())
    S = len(scenes)

    # 1. Per-scene bootstrap for each algorithm
    scene_boots = {alg: {} for alg in algs}
    for alg in algs:
        for s in scenes:
            scene_boots[alg][s] = bootstrap_scene_tiles(scene_tile_mccs[alg][s], B=B_scene)

    # 2. Global paired bootstrap
    alg_global = {alg: np.zeros(B_global) for alg in algs}
    for b in tqdm(range(B_global), desc="Global bootstrap"):
        idxs = np.random.randint(0, S, size=S)
        for alg in algs:
            vals = [scene_boots[alg][scenes[i]][np.random.randint(0, B_scene)] for i in idxs]
            alg_global[alg][b] = np.nanmedian(vals)

    # Paired differences
    diffs = {}
    for i in range(len(algs)):
        for j in range(i+1, len(algs)):
            a, bname = algs[i], algs[j]
            diffs[f"{a}_minus_{bname}"] = alg_global[a] - alg_global[bname]

    return pd.DataFrame(alg_global), pd.DataFrame(diffs)


def summarize(df):
    rows = []
    for col in df.columns:
        med = np.nanmedian(df[col])
        low, high = np.nanpercentile(df[col], [2.5, 97.5])
        rows.append({"metric": col, "median": med, "ci_lower": low, "ci_upper": high})
    return pd.DataFrame(rows)