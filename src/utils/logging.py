from pathlib import Path
from datetime import datetime
import logging

def setup_logging(
    log_dir= "results/logs",
    level=logging.INFO,
    max_logs=15
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove old logs
    logs = sorted(log_dir.glob("run_*.log"))
    if len(logs) > max_logs:
        for old_log in logs[:-max_logs]:
            old_log.unlink()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

    logging.info("Logging initialized")
    logging.info(f"Log file: {log_file}")
    return log_file
