from src.utils.plotting import save_figure

import os
import logging
from pathlib import Path
import rasterio # type: ignore
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns   # type: ignore
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    jaccard_score, matthews_corrcoef
    )

logger = logging.getLogger(__name__)


def compute_metrics(masks_dir: Path) -> pd.DataFrame:
    """Compute evaluation metrics for cloud masks in the given directory.
    Params:
        masks_dir: str, path to the directory containing 'reference' and algorithm subdirectories
    Returns: 
        pd.DataFrame with computed metrics for each scene and algorithm
    Raises:
        FileNotFoundError: If required directories or files are missing
    """
    ref_dir = os.path.join(masks_dir, 'reference')
    # Ensure reference directory exists.
    Path(ref_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Reference masks directory: {ref_dir}")

    # List of algorithm folders (each subfolder in 'masks' except 'reference').
    algorithms = [d for d in os.listdir(masks_dir) 
                if os.path.isdir(os.path.join(masks_dir, d)) and d != 'reference']
    logger.info(f"Algorithms found for evaluation: {', '.join(algorithms)}")

    records = []  # Collect per-scene results.
    for alg in algorithms:
        alg_dir = os.path.join(masks_dir, alg)
        # First, check if the algorithm directory exists and is not empty.
        if not os.path.exists(alg_dir):
            logger.warning(f"No algorithm folders found in masks directory: {masks_dir}")
            raise FileNotFoundError(f"No algorithm folders found in masks directory: {masks_dir}")
        if not os.listdir(alg_dir):
            logger.warning(f"Algorithm folder is empty: {alg_dir}")
            raise FileNotFoundError(f"Algorithm folder is empty: {alg_dir}")
        # Sort to ensure comparison with reference masks is correct.
        for file_name in sorted(os.listdir(alg_dir)):
            if not file_name.lower().endswith('.tif'):
                continue
            logger.info(f"Evaluating algorithm '{alg}' on file: {file_name}")

            scene_id = os.path.splitext(file_name)[0]
            ref_path = os.path.join(ref_dir, f"{scene_id}.tif")
            alg_path = os.path.join(alg_dir, file_name)

            # Load and flatten raster files for both reference and algorithm masks.
            with rasterio.open(ref_path) as ref_src, rasterio.open(alg_path) as alg_src:
                ref_arr = ref_src.read().flatten()
                alg_arr = alg_src.read().flatten()
                logger.debug(f"Reference mask shape: {ref_arr.shape}, Algorithm mask shape: {alg_arr.shape}")

            cm = confusion_matrix(ref_arr, alg_arr)

            # Flatten arrays (view of original, more memory efficient than flatten()).
            tn, fp, fn, tp = cm.ravel()
            logger.info(
                f"Confusion Matrix for scene '{scene_id}', \
                algorithm '{alg}': TP={tp}, TN={tn}, FP={fp}, FN={fn}")

            records.append({
                'scene_id': scene_id,
                'algorithm': alg,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'balanced_accuracy': balanced_accuracy_score(ref_arr, alg_arr),
                'precision': precision_score(ref_arr, alg_arr),   # Sensitivity
                'recall': recall_score(ref_arr, alg_arr),   # Specificity
                'f1_score': f1_score(ref_arr, alg_arr),
                'iou': jaccard_score(ref_arr, alg_arr),
                'mcc': matthews_corrcoef(ref_arr, alg_arr),
                'FPR': fp / (fp + tn),
                'FNR': fn / (fn + tp),
                'cloud_fraction': (tp + fp) / (tp + tn + fp + fn)
            })

    df = pd.DataFrame(records)

    logger.info("Completed computation of evaluation metrics.")
    logger.debug(f"Metrics DataFrame:\n{df.head()}")
    logger.debug(f"DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
    return df

def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    """Plot confusion matrix with counts and percentages.
    params:
        cm: confusion matrix (2D numpy array)
        title: str, title for the plot
    """
    # Check that confusion matrix array is 2x2 and non-empty.
    if cm.shape != (2, 2):
        logger.error(f"Confusion matrix must be 2x2. Received shape: {cm.shape}")
        raise ValueError("Confusion matrix must be 2x2.")
    if np.sum(cm) == 0:
        logger.error("Confusion matrix is empty (all zeros). Cannot plot.")
        raise ValueError("Confusion matrix is empty (all zeros). Cannot plot.")
    cm_percent = cm / cm.sum() * 100
    labels = np.asarray([
        [f"{count:,}\n({percent:.2f}%)" for count, percent in zip(row, row_p)]
        for row, row_p in zip(cm, cm_percent)
    ])
    logger.info(f"Plotting confusion matrix:\n{cm}\nPercentages:\n{cm_percent}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues',
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Must use .gcf() to get current figure for saving and avoid a TypeError.
    save_figure(plt.gcf(), Path(f"results/matrices/{title.replace(' ', '_').lower()}.png"))
    logger.info(f"Confusion matrix plot saved for: {title}")