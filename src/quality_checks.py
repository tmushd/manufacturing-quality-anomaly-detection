from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.transformations import sensor_columns


@dataclass
class CheckResult:
    check_name: str
    check_status: str
    failed_count: int
    details: str
    check_ts: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "check_status": self.check_status,
            "failed_count": self.failed_count,
            "details": self.details,
            "check_ts": self.check_ts,
        }


def _make_result(check_name: str, failed_count: int, details: str) -> CheckResult:
    return CheckResult(
        check_name=check_name,
        check_status="PASS" if failed_count == 0 else "FAIL",
        failed_count=int(failed_count),
        details=details,
        check_ts=datetime.now(timezone.utc),
    )


def run_row_count_check(bronze_features_df: pd.DataFrame, bronze_labels_df: pd.DataFrame) -> CheckResult:
    diff = abs(len(bronze_features_df) - len(bronze_labels_df))
    details = f"features={len(bronze_features_df)}, labels={len(bronze_labels_df)}"
    return _make_result("row_count_reconciliation", diff, details)


def run_schema_integrity_check(silver_clean_df: pd.DataFrame) -> CheckResult:
    sensors = sensor_columns(silver_clean_df)
    feature_count_failures = 0 if len(sensors) >= 1 else 1
    invalid_dtype_cols = [c for c in sensors if not np.issubdtype(silver_clean_df[c].dtype, np.number)]
    failed = feature_count_failures + len(invalid_dtype_cols)
    details = f"sensor_columns={len(sensors)}, invalid_dtype_cols={invalid_dtype_cols[:10]}"
    return _make_result("schema_integrity", failed, details)


def run_required_null_check(silver_clean_df: pd.DataFrame) -> CheckResult:
    invalid = silver_clean_df["row_id"].isna() | silver_clean_df["quality_label_binary"].isna()
    failed = int(invalid.sum())
    return _make_result("required_null_check", failed, "row_id and quality_label_binary must be non-null")


def run_label_integrity_check(labels_typed_df: pd.DataFrame) -> CheckResult:
    raw_invalid = ~labels_typed_df["quality_label_raw"].isin([-1, 1])
    binary_invalid = ~labels_typed_df["quality_label_binary"].isin([0, 1])
    failed = int(raw_invalid.sum() + binary_invalid.sum())
    details = f"raw_invalid={int(raw_invalid.sum())}, binary_invalid={int(binary_invalid.sum())}"
    return _make_result("label_integrity", failed, details)


def run_duplicate_check(silver_clean_df: pd.DataFrame) -> CheckResult:
    dupes = silver_clean_df.duplicated(subset=["row_id"]).sum()
    return _make_result("duplicate_row_id_check", int(dupes), "Duplicate row_id values in silver clean table")


def run_cast_failure_check(features_raw_df: pd.DataFrame, features_typed_df: pd.DataFrame) -> CheckResult:
    failed_count = 0
    sensors = sensor_columns(features_typed_df)
    for col in sensors:
        raw_not_null = features_raw_df[col].notna()
        typed_is_null = features_typed_df[col].isna()
        failed_count += int((raw_not_null & typed_is_null).sum())
    return _make_result("cast_failure_tracking", failed_count, "Raw non-null values that became null during numeric cast")


def run_null_threshold_compliance_check(features_typed_df: pd.DataFrame, threshold: float = config.NULL_DROP_THRESHOLD) -> CheckResult:
    sensors = sensor_columns(features_typed_df)
    ratio_violations = 0
    for col in sensors:
        ratio = float(features_typed_df[col].isna().mean())
        if ratio > threshold:
            ratio_violations += 1
    details = f"threshold={threshold}, violating_columns={ratio_violations}"
    return _make_result("null_threshold_compliance", ratio_violations, details)


def run_gold_scoring_completeness_check(predictions_df: pd.DataFrame) -> CheckResult:
    missing = predictions_df["anomaly_score"].isna() | predictions_df["anomaly_flag"].isna()
    failed = int(missing.sum())
    return _make_result("gold_scoring_completeness", failed, "anomaly_score and anomaly_flag must be non-null")


def run_all_checks(
    bronze_features_df: pd.DataFrame,
    bronze_labels_df: pd.DataFrame,
    silver_features_typed_df: pd.DataFrame,
    silver_labels_typed_df: pd.DataFrame,
    silver_clean_df: pd.DataFrame,
    gold_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        run_row_count_check(bronze_features_df, bronze_labels_df),
        run_schema_integrity_check(silver_clean_df),
        run_required_null_check(silver_clean_df),
        run_label_integrity_check(silver_labels_typed_df),
        run_duplicate_check(silver_clean_df),
        run_cast_failure_check(bronze_features_df, silver_features_typed_df),
        run_null_threshold_compliance_check(silver_features_typed_df),
        run_gold_scoring_completeness_check(gold_predictions_df),
    ]
    return pd.DataFrame([check.as_dict() for check in checks])


def persist_check_results(results_df: pd.DataFrame) -> None:
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(config.GOLD_DIR / "data_quality_results.parquet", index=False)
    results_df.to_csv(config.GOLD_DIR / "data_quality_results.csv", index=False)
