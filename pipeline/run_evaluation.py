"""
Run evaluation metrics and confusion matrix analysis.
"""
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix  # type: ignore

from src.analysis.evaluation import compute_metrics, plot_confusion_matrix
from src.utils.io import save_csv, save_dataframe, validate_output_path, load_masks
from src.utils.validator import DataValidator
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run(
    algorithms: list[str],
    masks_dir: Path,
    reference_masks_dir: Path,
    data_dir: Path,
    validation_config: DataValidator,
) -> Path:
    """
    Execute evaluation pipeline.
    Args:
        algorithms: List of algorithm names to evaluate.
        masks_dir: Path to directory containing algorithm mask subdirectories.
        reference_masks_dir: Path to directory containing reference masks.
        data_dir: Path to directory for saving output metrics and plots.
        validation_config: DataValidator instance for validating computed metrics.
    Returns:
        output_csv: Path to saved metrics CSV.
    """
    logger.info("Starting evaluation process.")

    output_csv = data_dir / "per_scene_evaluation_metrics.csv"
    validate_output_path(output_csv, required_space_mb=10)

    # Compute evaluation metrics
    df = compute_metrics(masks_dir)

    # Load reference masks once to avoid redundant I/O operations during metric computation.
    reference_masks_list = load_masks(reference_masks_dir)

    # logger.debug(f"Loaded {len(reference_masks_list)} reference masks for evaluation.")
    # Compute and plot confusion matrices for each scene and algorithm combination.
    for alg in algorithms:
        alg_masks_list = load_masks(masks_dir / alg)

        for i, (ref_mask, alg_mask) in enumerate(zip(reference_masks_list, alg_masks_list)):
            cm = confusion_matrix(ref_mask, alg_mask)
            plot_confusion_matrix(cm, title=f"{alg} Confusion Matrix Scene {i+1}")
            logger.info(f"Plotted confusion matrix for algorithm '{alg}', scene {i+1}")

    # Strictly validate the computed metrics dataframe before saving.
    validator = validation_config
    validator.validate_strict(df, context="metrics pre-save")

    # Save temporarily using atomic save to ensure data integrity.
    save_dataframe(df, output_csv)
    # Save evaluation metrics to CSV in the data directory to use for analysis.
    save_csv(df, output_csv)
    logger.info("Evaluation completed successfully.")
    return output_csv

def cli() -> None:
    setup_logging()

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
        reference_masks_dir = Path(config["paths"]["reference_masks_dir"])
    except KeyError as e:
        raise SystemExit(f"Missing required configuration key: {e}")
    
    validation_config = DataValidator(
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

    output_path = run(
        algorithms=algorithms,
        masks_dir=masks_dir,
        reference_masks_dir=reference_masks_dir,
        data_dir=data_dir,
        validation_config=validation_config,
    )

    print(f"Evaluation completed. Metrics saved to: {output_path}")

if __name__ == "__main__":
    cli()