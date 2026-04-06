from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.transformations import sensor_columns


def select_model_features(clean_df: pd.DataFrame) -> list[str]:
    return sensor_columns(clean_df)


def scale_features(clean_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(clean_df[feature_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=feature_cols, index=clean_df.index)
    return scaled_df, scaler


def prepare_training_matrix(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = select_model_features(clean_df)
    x_df, _ = scale_features(clean_df, feature_cols)
    y = clean_df["quality_label_binary"].astype(int)
    return x_df, y, feature_cols
