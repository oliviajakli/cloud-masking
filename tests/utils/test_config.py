import pytest
import yaml

from src.utils.config import load_config


def test_load_config_success(tmp_path):
    cfg = {
        "paths": {
            "data_root": "data",
            "reference_masks_dir": "data/ref",
            "output_dir": "results"
        },
        "statistics": {"random_seed": 42}
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    loaded = load_config(str(cfg_path))
    assert loaded["paths"]["data_root"] == "data"
    assert loaded["statistics"]["random_seed"] == 42


def test_load_config_missing_raises(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(str(missing))
