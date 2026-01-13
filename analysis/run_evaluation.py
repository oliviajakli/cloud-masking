from src.evaluation import compute_metrics, plot_confusion_matrix
from src.utils.config import load_config
from pathlib import Path
from sklearn.metrics import confusion_matrix

config = load_config('config.yaml')

input_data = Path(config["paths"]["input"])
algorithms = config["algorithms"]
output_dir = Path(config["paths"]["output_dir"])

masks_dir = Path(config["data_root"]) / "masks"
reference_masks = Path(config["reference_masks_dir"])

def main():
    # Run evaluation metrics, plot confusion matrices, and save results.
    compute_metrics(masks_dir)
    for alg in algorithms:
        cm = confusion_matrix(reference_masks, Path(config[f"{alg}_masks_dir"]))
        plot_confusion_matrix(cm, title=f"{alg} Confusion Matrix")
    return "Evaluation completed. Results saved to:", output_dir

if __name__ == "__main__":
    print(main())