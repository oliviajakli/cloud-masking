from pathlib import Path
import pandas as pd
from datetime import datetime

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_csv(df: pd.DataFrame, path: Path, timestamp=True):
    ensure_dir(path.parent)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path.with_stem(f"{path.stem}_{ts}")
    df.to_csv(path, index=False)