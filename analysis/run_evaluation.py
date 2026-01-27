from src.evaluation import load_masks, compute_metrics, plot_confusion_matrix
from src.utils.config import load_config
from src.utils.io import save_csv
from pathlib import Path
import os
import logging
from sklearn.metrics import confusion_matrix    # type: ignore

from src.utils.logging import setup_logging    # type: ignore

logger = logging.getLogger(__name__)

config = load_config()

algorithms = config["algorithms"]
masks_dir = Path(config["paths"]["data_root"]) / "masks"
reference_masks = Path(config["paths"]["reference_masks_dir"])

def main() -> tuple[str, str]:
    """Run evaluation metrics, plot confusion matrices, and save results.
    Returns:
        message: str, status message
        output_csv: str, path to saved CSV file with evaluation metrics
    """
    setup_logging()
    logger.info("Starting evaluation process...")
    df = compute_metrics(masks_dir)
    reference_masks_list = load_masks(reference_masks)
    logger.debug(f"Loaded {len(reference_masks_list)} reference masks for evaluation.")
    for alg in algorithms:
        alg_masks_list = load_masks(Path(config["paths"][f"{alg}_masks_dir"]))
        for i, (ref_mask, alg_mask) in enumerate(zip(reference_masks_list, alg_masks_list)):
            cm = confusion_matrix(ref_mask, alg_mask)
            plot_confusion_matrix(cm, title=f"{alg} Confusion Matrix Scene {i+1}")
            logger.info(f"Plotted confusion matrix for algorithm '{alg}', scene {i+1}")
    # Save evaluation metrics to CSV in the data directory to use for analysis.
    output_csv = os.path.join('data', 'per_scene_evaluation_metrics.csv')
    save_csv(df, Path(output_csv))
    return "Evaluation completed. Metrics saved to:", output_csv

if __name__ == "__main__":
    message, path = main()
    print(message, path)