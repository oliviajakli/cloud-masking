import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import rasterio  # type: ignore

logger = logging.getLogger(__name__)

# Make directories as needed.
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# Save DataFrame to CSV with optional timestamp in filename.
def save_csv(df: pd.DataFrame, path: Path, timestamp=False) -> None:
    """Save DataFrame to CSV, optionally appending a timestamp to the filename.
    Args:
        df: DataFrame to save
        path: Path to save CSV file
        timestamp: Whether to append timestamp to filename
    Returns:
        None
    """
    ensure_dir(path.parent)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path.with_stem(f"{path.stem}_{ts}")
    df.to_csv(path, index=False)

# Validate the user-specified output path early.
def validate_output_path(filepath: Path, required_space_mb:float=100.0):
    """
    Validate that we can write to the specified path.
    
    Args:
        filepath: Path where CSV will be written
        required_space_mb: Estimated space needed in MB (default 100MB)
    
    Raises:
        ValueError: If path validation fails
    """
    path = Path(filepath)
    
    # Check if parent directory exists.
    parent_dir = path.parent
    if not parent_dir.exists():
        raise ValueError(
            f"Directory does not exist: {parent_dir}. "
            f"Please create it first or choose a different location."
        )
    
    # Check if parent directory is actually a directory.
    if not parent_dir.is_dir():
        raise ValueError(
            f"Parent path is not a directory: {parent_dir}"
        )
    
    # Check write permissions by attempting to create a temp file.
    try:
        with tempfile.NamedTemporaryFile(dir=parent_dir, delete=True):
            pass
    except PermissionError:
        raise ValueError(
            f"Permission denied: Cannot write to directory {parent_dir}. "
            f"Please check permissions or choose a different location."
        )
    except OSError as e:
        raise ValueError(
            f"Cannot write to directory {parent_dir}: {e}"
        )
    
    # Check available disk space.
    stat = shutil.disk_usage(parent_dir)
    available_mb = stat.free / (1024 * 1024)
    
    if available_mb < required_space_mb:
        raise ValueError(
            f"Insufficient disk space: {available_mb:.1f}MB available, "
            f"but {required_space_mb}MB recommended. "
            f"Please free up space or choose a different location."
        )
    
    # Warn if file already exists (optional - you might want to allow overwriting)
    if path.exists():
        logger.warning(f"Warning: {filepath} already exists and will be overwritten.")
    
    return True

# Estimate CSV size from DataFrame memory usage.
def estimate_csv_size(df: pd.DataFrame) -> float:
    """
    Estimate CSV file size in MB.
    Returns a conservative estimate (actual size may be smaller).
    """
    # Quick estimate: memory usage * 1.5 (CSV is often larger than in-memory)
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    estimated_csv_mb = memory_usage_mb * 1.5
    return estimated_csv_mb


# More sophisticated validation.
def validate_output_path_for_df(filepath: Path, df: pd.DataFrame, buffer_mb:int=20):
    """Validate path with size estimate from actual DataFrame
    Args:
        filepath: Path to validate
        df: DataFrame to be saved
        buffer_mb: Additional buffer space in MB
    Raises:
        ValueError: If validation fails
    """
    estimated_size = estimate_csv_size(df)
    required_space = estimated_size + buffer_mb  # Add buffer
    validate_output_path(filepath, required_space_mb=required_space)

# Save DataFrame for critical evaluation metrics (pipeline dependency).
def save_dataframe(df: pd.DataFrame, filepath: Path):
    """Save DataFrame to CSV atomically, raising on failure.
    Args:
        df: DataFrame to save
        filepath: Path to save CSV file
    Raises:
        RuntimeError: If saving fails"""    
    try:
        # Write to temporary file first
        temp_dir = Path(filepath).parent
        with tempfile.NamedTemporaryFile(
            mode='w', 
            delete=False, 
            dir=temp_dir,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = tmp_file.name
            df.to_csv(tmp_file, index=False)
        
        # Atomic rename only if write succeeded
        shutil.move(tmp_path, filepath)
        
    except OSError as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"CRITICAL: Failed to save evaluation metrics to {filepath}. "
            f"Pipeline cannot continue. Error: {e}"
        ) from e

# For user-facing analysis results.
def save_analysis_results(df: pd.DataFrame, filepath: Path, buffer_mb:int=20):
    """Save analysis results DataFrame to CSV with user-friendly error handling.
    Args:
        df: DataFrame to save
        filepath: Path to save CSV file
        buffer_mb: Additional buffer space in MB used for path validation.
    Raises:
        RuntimeError: If saving fails due to permission or disk space issues.
    """
    ensure_dir(filepath.parent)
    # Validate writeability and available space using the actual dataframe size.
    validate_output_path_for_df(filepath, df, buffer_mb=buffer_mb)
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
        
    except PermissionError:
        raise RuntimeError(
            f"Permission denied: Cannot write to {filepath}. "
            f"Please check file permissions or choose a different location."
        ) from None
        
    except OSError as e:
        if e.errno == 28:  # ENOSPC - No space left on device
            raise RuntimeError(
                f"Disk full: Cannot save results to {filepath}. "
                f"Please free up space or choose a different location."
            ) from None
        raise RuntimeError(
            f"Failed to save results to {filepath}: {e}"
        ) from e

def load_masks(folder_path: Path) -> list[np.ndarray]:
    """Load all mask file paths from a given folder.
    params:
        folder_path: str, path to folder containing mask files
    returns: list of numpy arrays
    """
    masks = []
    # Must be sorted to ensure comparison with reference masks is correct.
    for file in sorted(os.listdir(folder_path)):
        logger.info(f"Loading mask file: {file}")
        # Check that folder is not empty.
        if len(file) == 0:
            logger.warning(f"No files found in folder: {folder_path}")
            raise FileNotFoundError(f"No files found in folder: {folder_path}")
        # Check that file is not empty.
        path = os.path.join(folder_path, file)
        if Path(path).stat().st_size == 0:
            logger.error(f"File is empty: {path}")
            raise ValueError(f"File is empty: {path}")
        # Only process .tif files.
        if not file.lower().endswith('.tif'):
            continue
        with rasterio.open(os.path.join(folder_path, file)) as src:
            mask = src.read()
            logger.debug(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            masks.append(mask.flatten())    # 1D array shape for metric computations.
            logger.info(f"Loaded mask shape: {mask.shape}")
    return masks