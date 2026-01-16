import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    jaccard_score, matthews_corrcoef
    )

from pathlib import Path

from src.utils.plotting import save_figure

def load_masks(folder_path):
    """Load all mask file paths from a given folder."""
    masks = []
    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith('.tif'):
            continue
        with rasterio.open(os.path.join(folder_path, file)) as src:
            mask = src.read()
            masks.append(mask.flatten())
    return masks

def compute_metrics(masks_dir):
    # Define directories
    ref_dir = os.path.join(masks_dir, 'reference')

    # List of algorithm folders (each subfolder in 'masks' except 'reference')
    algorithms = [d for d in os.listdir(masks_dir) 
                if os.path.isdir(os.path.join(masks_dir, d)) and d != 'reference']

    records = []  # list to collect per-scene results

    # Loop through algorithms and scenes
    for alg in algorithms:
        alg_dir = os.path.join(masks_dir, alg)
        for file_name in sorted(os.listdir(alg_dir)):
            if not file_name.lower().endswith('.tif'):
                continue

            scene_id = os.path.splitext(file_name)[0]
            ref_path = os.path.join(ref_dir, f"{scene_id}.tif")
            alg_path = os.path.join(alg_dir, file_name)

            # Load both rasters
            with rasterio.open(ref_path) as ref_src, rasterio.open(alg_path) as alg_src:
                ref_arr = ref_src.read().flatten()
                alg_arr = alg_src.read().flatten()

            # Flatten arrays and compute confusion matrix
            cm = confusion_matrix(ref_arr, alg_arr)
            tn, fp, fn, tp = cm.ravel()

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
                'mcc': matthews_corrcoef(ref_arr, alg_arr)
            })

    # Create DataFrame
    df = pd.DataFrame(records)

    # Compute false positive and false negatives rates.
    df['FPR'] = df['FP'] / (df['FP'] + df['TN'])
    df['FNR'] = df['FN'] / (df['FN'] + df['TP'])
    # Calculate cloud fraction (proportion of cloud pixels to total pixels) per scene.
    df["cloud_fraction"] = (df["TP"] + df["FP"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"])
    return df

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix with counts and percentages."""
    cm_percent = cm / cm.sum() * 100
    labels = np.asarray([
        [f"{count:,}\n({percent:.2f}%)" for count, percent in zip(row, row_p)]
        for row, row_p in zip(cm, cm_percent)
    ])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues',
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_figure(plt.gcf(), Path(f"results/matrices/{title.replace(' ', '_').lower()}.png"))