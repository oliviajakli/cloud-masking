from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataValidator:
    # Schema
    required_columns: Set[str]
    metric_columns: Set[str]

    # Expectations
    expected_algorithms: Optional[Set[str]] = None

    # Value rules (used in strict mode)
    value_constraints: Dict[str, Callable[[pd.Series], pd.Series]] = field(
        default_factory=dict
    )

    # ---------- PUBLIC API ----------
    def validate_strict(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Strict validation for freshly computed, in-memory DataFrames.
        Fail-fast. Any failure is a bug.
        """
        self._check_empty(df, context)
        self._validate_schema(df, context)
        self._validate_uniqueness(df, context)
        self._validate_algorithms(df, strict=True, context=context)
        self._validate_metrics_strict(df, context)

        logger.info("Strict validation passed%s", self._ctx(context))
        return df

    def validate_light(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Lightweight validation for CSVs loaded later in the pipeline.
        Cheap, schema-focused, safe to call often.
        """
        self._check_empty(df, context)
        self._validate_schema(df, context)
        self._validate_algorithms(df, strict=False, context=context)
        self._validate_metrics_light(df, context)

        logger.debug("Light validation passed%s", self._ctx(context))
        return df

    # ---------- SHARED CHECKS ----------
    def _check_empty(self, df: pd.DataFrame, context: str):
        if df.empty:
            raise ValueError(f"DataFrame is empty{self._ctx(context)}")

    def _validate_schema(self, df: pd.DataFrame, context: str):
        missing = self.required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns {missing}{self._ctx(context)}"
            )

    def _validate_algorithms(
        self, df: pd.DataFrame, *, strict: bool, context: str
    ):
        if self.expected_algorithms is None:
            return

        observed = set(df["algorithm"].dropna().unique())
        unexpected = observed - self.expected_algorithms

        if unexpected:
            raise ValueError(
                f"Unexpected algorithm values {unexpected}{self._ctx(context)}"
            )

        if strict and observed != self.expected_algorithms:
            raise ValueError(
                f"Algorithm set mismatch. "
                f"Expected {self.expected_algorithms}, got {observed}"
                f"{self._ctx(context)}"
            )

    # ---------- STRICT-ONLY ----------
    def _validate_uniqueness(self, df: pd.DataFrame, context: str):
        if df.duplicated(subset=["algorithm", "sample_id"]).any():
            raise ValueError(
                f"Duplicate (algorithm, sample_id) rows detected"
                f"{self._ctx(context)}"
            )

    def _validate_metrics_strict(self, df: pd.DataFrame, context: str):
        metrics = df[list(self.metric_columns)]

        if not np.isfinite(metrics.to_numpy()).all():
            raise ValueError(
                f"Non-finite metric values detected{self._ctx(context)}"
            )

        for col, constraint in self.value_constraints.items():
            mask = constraint(df[col])
            if not mask.all():
                bad = (~mask).sum()
                raise ValueError(
                    f"Column '{col}' has {bad} invalid values"
                    f"{self._ctx(context)}"
                )

    # ---------- LIGHT-ONLY ----------
    def _validate_metrics_light(self, df: pd.DataFrame, context: str):
        metrics = df[list(self.metric_columns)]

        # Catch silent corruption
        if metrics.isna().all(axis=1).any():
            raise ValueError(
                f"One or more rows contain only NaN metrics"
                f"{self._ctx(context)}"
            )

    # ---------- helpers ----------
    @staticmethod
    def _ctx(context: str) -> str:
        return f" [{context}]" if context else ""