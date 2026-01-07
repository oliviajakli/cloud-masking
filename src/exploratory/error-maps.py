import itertools
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from src.config_loader import load_config

config = load_config()

input_data = Path(config["paths"]["input"])
algorithms = config["algorithms"]
samples = config["samples"]
reference_masks = Path(config["paths"]["reference_dir"])
output_dir = Path(config["paths"]["output_dir"])

for alg, sample in itertools.product(algorithms, samples):
    reference_path = f"{reference_masks}/{sample}.tif"
    predicted_path = f"{Path(config["paths"][f"{alg}_dir"])}/{sample}.tif"
    out_path = f"{output_dir}/error_map_{alg}_{sample}.tif"

    with rasterio.open(reference_path) as ref_ds, rasterio.open(predicted_path) as pred_ds:
        reference = ref_ds.read(1)
        predicted = pred_ds.read(1)

    error_map = np.zeros_like(reference, dtype=np.uint8)
    error_map[(reference == 1) & (predicted == 1)] = 1
    error_map[(reference == 0) & (predicted == 0)] = 2
    error_map[(reference == 0) & (predicted == 1)] = 3
    error_map[(reference == 1) & (predicted == 0)] = 4

    colors = ["black", "lime", "gray", "red", "orange"]  # BG, TP, TN, FP, FN
    labels = ["Background", "True Positive", "True Negative", "False Positive", "False Negative"]
    cmap = ListedColormap(colors[1:])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(error_map, cmap=cmap, interpolation="none")
    ax.set_title(f"Error Map – {alg.upper()} ({sample})", fontsize=14)
    ax.axis("off")

    patches = [mpatches.Patch(color=colors[i+1], label=labels[i+1]) for i in range(4)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

print(f"Per-pixel error maps saved in {output_dir} folder.")