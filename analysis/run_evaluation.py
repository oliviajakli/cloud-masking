from src.evaluation import load_masks, compute_metrics, plot_confusion_matrix
from src.utils.config import load_config
from src.utils.io import save_csv
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix

config = load_config()

input_data = Path(config["paths"]["input"])
algorithms = config["algorithms"]

masks_dir = Path(config["paths"]["data_root"]) / "masks"
reference_masks = Path(config["paths"]["reference_masks_dir"])

def main():
    # Run evaluation metrics, plot confusion matrices, and save results.
    df = compute_metrics(masks_dir)
    reference_masks_list = load_masks(reference_masks)
    for alg in algorithms:
        alg_masks_list = load_masks(Path(config["paths"][f"{alg}_masks_dir"]))
        for i, (ref_mask, alg_mask) in enumerate(zip(reference_masks_list, alg_masks_list)):
            cm = confusion_matrix(ref_mask, alg_mask)
            plot_confusion_matrix(cm, title=f"{alg} Confusion Matrix Scene {i+1}")
    # Save to CSV
    output_csv = os.path.join('data', 'per_scene_evaluation_metrics.csv')
    save_csv(df, Path(output_csv))
    return "Evaluation completed. Metrics saved to:", output_csv

if __name__ == "__main__":
    print(main())