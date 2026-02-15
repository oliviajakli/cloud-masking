import pandas as pd
import pytest

from src.utils.io import save_csv, validate_output_path


def test_save_csv_and_read(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = tmp_path / "out.csv"

    # ensure no exception is raised and file is created
    save_csv(df, out, timestamp=False)
    read = pd.read_csv(out)
    assert list(read.columns) == ["a"]
    assert read.shape == (3, 1)

def test_validate_output_path_missing_parent(tmp_path):
    # parent does not exist
    target = tmp_path / "no_such_dir" / "file.csv"
    with pytest.raises(ValueError):
        validate_output_path(target)
