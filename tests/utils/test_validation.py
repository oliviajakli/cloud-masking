import pandas as pd
import pytest

"""Strict validation tests"""

# Passes on valid input.
def test_validate_strict_passes(valid_df, validator):
    result = validator.validate_strict(valid_df)
    assert result.equals(valid_df)

# Fails on empty DataFrame.
def test_validate_strict_empty_df(validator):
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validator.validate_strict(df)

# Fails on missing required columns.
def test_validate_strict_missing_column(valid_df, validator):
    df = valid_df.drop(columns=["precision"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validator.validate_strict(df)

# Fails on duplicate rows.
def test_validate_strict_duplicate_rows(valid_df, validator):
    df = pd.concat([valid_df, valid_df.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate"):
        validator.validate_strict(df)

# Fails on unexpected algorithm.
def test_validate_strict_unexpected_algorithm(valid_df, validator):
    df = valid_df.copy()
    df.loc[0, "algorithm"] = "C"

    with pytest.raises(ValueError, match="Unexpected algorithm"):
        validator.validate_strict(df)

# Fails on algorithm set mismatch.
def test_validate_strict_algorithm_set_mismatch(valid_df, validator):
    df = valid_df[valid_df["algorithm"] == "A"]

    with pytest.raises(ValueError, match="Algorithm set mismatch"):
        validator.validate_strict(df)

# Fails on non-finite values.
def test_validate_strict_non_finite(valid_df, validator):
    df = valid_df.copy()
    df.loc[0, "precision"] = float("inf")

    with pytest.raises(ValueError, match="Non-finite"):
        validator.validate_strict(df)

# Fails on constraint violation.
def test_validate_strict_constraint_violation(valid_df, validator):
    df = valid_df.copy()
    df.loc[0, "precision"] = 1.5

    with pytest.raises(ValueError, match="invalid values"):
        validator.validate_strict(df)

"""Light validation tests"""

# Light mode should not enforce exact algorithm set.
def test_validate_light_allows_missing_algorithm(valid_df, validator):
    df = valid_df[valid_df["algorithm"] == "A"]
    validator.validate_light(df)

# Fails if all metric values are NaN (indicates something is wrong with the data).
def test_validate_light_all_nan_metrics(valid_df, validator):
    df = valid_df.copy()
    df[["precision", "recall"]] = float("nan")

    with pytest.raises(ValueError, match="only NaN metrics"):
        validator.validate_light(df)

"""Test _validate_no_new_nans, even though it's private."""

def test_validate_no_new_nans(valid_df, validator):
    ref_counts = {
        "precision": valid_df["precision"].notna().sum()
    }

    df = valid_df.copy()
    df.loc[0, "precision"] = float("nan")

    with pytest.raises(ValueError, match="lost"):
        validator._validate_no_new_nans(
            df,
            reference_non_nulls=ref_counts,
            context="test"
        )