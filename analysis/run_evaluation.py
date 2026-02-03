from src.evaluation import load_masks, compute_metrics, plot_confusion_matrix
from src.utils.config import load_config
from src.utils.io import save_csv, save_dataframe, validate_output_path
from src.utils.logging import setup_logging
from src.utils.validator import DataValidator    # type: ignore

import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix    # type: ignore


logger = logging.getLogger(__name__)

# Load and validate configuration
try:
    config = load_config()
except FileNotFoundError as e:
    raise SystemExit(f"Configuration file not found: {e}")
except Exception as e:
    raise SystemExit(f"Failed to load configuration: {e}")

try:
    algorithms = config["algorithms"]
    data_dir = Path(config["paths"]["data_root"])
    masks_dir = Path(config["paths"]["data_root"]) / "masks"
    reference_masks = Path(config["paths"]["reference_masks_dir"])
except KeyError as e:
    raise SystemExit(f"Missing required configuration key: {e}")

def main() -> tuple[str, Path]:
    """Run evaluation metrics, plot confusion matrices, and save results.
    Returns:
        message: str, status message
        output_csv: Path, path to saved CSV file with evaluation metrics
    """
    setup_logging()
    logger.info("Starting evaluation process...")
    # Set output path for evaluation metrics CSV.
    output_csv = Path(f"{data_dir}/per_scene_evaluation_metrics.csv")
    # Validate critical path upfront.
    validate_output_path(output_csv, required_space_mb=10)
    # Compute evaluation metrics DataFrame.
    df = compute_metrics(masks_dir)
    reference_masks_list = load_masks(reference_masks)
    logger.debug(f"Loaded {len(reference_masks_list)} reference masks for evaluation.")
    # Compute and plot confusion matrices for each scene and algorithm combination.
    for alg in algorithms:
        alg_masks_list = load_masks(Path(config["paths"][f"{alg}_masks_dir"]))
        for i, (ref_mask, alg_mask) in enumerate(zip(reference_masks_list, alg_masks_list)):
            cm = confusion_matrix(ref_mask, alg_mask)
            plot_confusion_matrix(cm, title=f"{alg} Confusion Matrix Scene {i+1}")
            logger.info(f"Plotted confusion matrix for algorithm '{alg}', scene {i+1}")
    
    validator = DataValidator(
        required_columns=set(config["validation"]["required_columns"]),
        metric_columns=set(config["validation"]["metric_columns"]),
        expected_algorithms=set(algorithms),
        value_constraints={
            "precision": lambda s: s.between(0, 1),
            "recall": lambda s: s.between(0, 1),
            "f1_score": lambda s: s.between(0, 1),
            "iou": lambda s: s.between(0, 1),
            "mcc": lambda s: s.between(-1, 1),
            "FPR": lambda s: s.between(0, 1),
            "FNR": lambda s: s.between(0, 1),
            "cloud_fraction": lambda s: s.between(0, 1)
        }
    )
    validator.validate_strict(df, context="metrics pre-save")
    # Save temporarily using atomic save to ensure data integrity.
    save_dataframe(df, output_csv)
    # Save evaluation metrics to CSV in the data directory to use for analysis.
    save_csv(df, output_csv)
    return "Evaluation completed. Metrics saved to:", data_dir

if __name__ == "__main__":
    message, path = main()
    print(message, path)