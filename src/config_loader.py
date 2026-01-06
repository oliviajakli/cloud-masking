import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Load configuration from a YAML file."""
    with open(Path(config_path), 'r') as file:
        return yaml.safe_load(file)