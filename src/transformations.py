from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger


logger = get_logger(__name__)


@dataclass
class SilverArtifacts:
    features_typed: pd.DataFrame
    labels_typed: pd.DataFrame
    manufacturing_clean: pd.DataFrame
    missingness_profile: pd.DataFrame
    dropped_high_null_cols: list[str]
    dropped_constant_cols: list[str]
    imputation_stats: pd.DataFrame


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def cast_sensor_columns(features_df: pd.DataFrame) -> pd.DataFrame:
    typed = features_df.copy()
    for col in sensor_columns(typed):
        typed[col] = pd.to_numeric(typed[col], errors="coerce")
    return typed


def build_missingness_profile(features_df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    total_rows = len(features_df)
    profile = []
    for col in cols:
        null_count = int(features_df[col].isna().sum())
        profile.append(
            {
                "column_name": col,
                "null_count": null_count,
                "null_ratio": (null_count / total_rows) if total_rows else 0.0,
            }
        )
    return pd.DataFrame(profile).sort_values("null_ratio", ascending=False)


def drop_high_null_columns(features_df: pd.DataFrame, cols: Iterable[str], threshold: float) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    profile = build_missingness_profile(features_df, cols)
    drop_cols = profile.loc[profile["null_ratio"] > threshold, "column_name"].tolist()
    kept_df = features_df.drop(columns=drop_cols)
    return kept_df, drop_cols, profile


def drop_constant_and_near_constant_columns(features_df: pd.DataFrame, cols: Iterable[str], near_constant_threshold: float = config.NEAR_CONSTANT_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    drop_cols: list[str] = []
    row_count = max(len(features_df), 1)
    for col in cols:
        col_series = features_df[col]
        if col_series.nunique(dropna=True) <= 1:
            drop_cols.append(col)
            continue

        top_freq = col_series.value_counts(dropna=True).iloc[0]
        if (top_freq / row_count) >= near_constant_threshold:
            drop_cols.append(col)

    return features_df.drop(columns=drop_cols), drop_cols


def impute_median(features_df: pd.DataFrame, cols: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filled = features_df.copy()
    stats = []
    for col in cols:
        median_value = filled[col].median(skipna=True)
        if pd.isna(median_value):
            median_value = 0.0
        missing_before = int(filled[col].isna().sum())
        filled[col] = filled[col].fillna(median_value)
        stats.append(
            {
                "column_name": col,
                "median_value": float(median_value),
                "missing_before": missing_before,
                "missing_after": int(filled[col].isna().sum()),
            }
        )
    return filled, pd.DataFrame(stats)


def standardize_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    typed = labels_df.copy()
    typed["quality_label_raw"] = pd.to_numeric(typed["raw_label"], errors="coerce").astype("Int64")
    typed["quality_label_binary"] = typed["quality_label_raw"].map({-1: 0, 1: 1}).astype("Int64")
    typed["event_ts"] = pd.to_datetime(typed["raw_timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce", utc=True)
    return typed


def join_features_and_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    label_subset = labels_df[["row_id", "quality_label_raw", "quality_label_binary", "event_ts"]]
    joined = features_df.merge(label_subset, on="row_id", how="inner", validate="one_to_one")
    joined = joined.drop_duplicates(subset=["row_id"])
    return joined


def build_silver_tables(features_raw_df: pd.DataFrame, labels_raw_df: pd.DataFrame) -> SilverArtifacts:
    typed_features = cast_sensor_columns(features_raw_df)
    all_sensor_cols = sensor_columns(typed_features)

    reduced_features, dropped_high_null_cols, missingness_profile = drop_high_null_columns(
        typed_features,
        all_sensor_cols,
        config.NULL_DROP_THRESHOLD,
    )

    post_null_sensor_cols = sensor_columns(reduced_features)
    reduced_features, dropped_constant_cols = drop_constant_and_near_constant_columns(
        reduced_features,
        post_null_sensor_cols,
    )

    post_constant_sensor_cols = sensor_columns(reduced_features)
    imputed_features, imputation_stats = impute_median(reduced_features, post_constant_sensor_cols)

    labels_typed = standardize_labels(labels_raw_df)
    clean_joined = join_features_and_labels(imputed_features, labels_typed)

    return SilverArtifacts(
        features_typed=imputed_features,
        labels_typed=labels_typed,
        manufacturing_clean=clean_joined,
        missingness_profile=missingness_profile,
        dropped_high_null_cols=dropped_high_null_cols,
        dropped_constant_cols=dropped_constant_cols,
        imputation_stats=imputation_stats,
    )


def write_silver_tables(artifacts: SilverArtifacts) -> dict[str, str]:
    config.SILVER_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "secom_features_typed": str(config.SILVER_DIR / "secom_features_typed.parquet"),
        "secom_labels_typed": str(config.SILVER_DIR / "secom_labels_typed.parquet"),
        "secom_manufacturing_clean": str(config.SILVER_DIR / "secom_manufacturing_clean.parquet"),
        "missingness_profile": str(config.SILVER_DIR / "missingness_profile.parquet"),
        "imputation_stats": str(config.SILVER_DIR / "imputation_stats.parquet"),
    }

    artifacts.features_typed.to_parquet(paths["secom_features_typed"], index=False)
    artifacts.labels_typed.to_parquet(paths["secom_labels_typed"], index=False)
    artifacts.manufacturing_clean.to_parquet(paths["secom_manufacturing_clean"], index=False)
    artifacts.missingness_profile.to_parquet(paths["missingness_profile"], index=False)
    artifacts.imputation_stats.to_parquet(paths["imputation_stats"], index=False)

    artifacts.features_typed.to_csv(config.SILVER_DIR / "secom_features_typed.csv", index=False)
    artifacts.labels_typed.to_csv(config.SILVER_DIR / "secom_labels_typed.csv", index=False)
    artifacts.manufacturing_clean.to_csv(config.SILVER_DIR / "secom_manufacturing_clean.csv", index=False)

    logger.info("Wrote silver tables to %s", config.SILVER_DIR)
    return paths
