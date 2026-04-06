from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import config
from src.utils import safe_divide


def split_data(x_df: pd.DataFrame, y: pd.Series, test_size: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        x_df,
        y,
        test_size=test_size,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def train_isolation_forest(x_train: pd.DataFrame, y_train: pd.Series | None = None, train_on_pass_only: bool = True) -> IsolationForest:
    model = IsolationForest(**asdict(config.MODEL_CONFIG))
    if train_on_pass_only and y_train is not None:
        train_mask = y_train == 0
        model.fit(x_train.loc[train_mask])
    else:
        model.fit(x_train)
    return model


def score_records(model: IsolationForest, x_df: pd.DataFrame, contamination: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    if contamination is None:
        contamination = config.MODEL_CONFIG.contamination

    anomaly_score = model.score_samples(x_df)
    threshold = float(np.quantile(anomaly_score, contamination))
    anomaly_flag = (anomaly_score <= threshold).astype(int)
    return anomaly_score, anomaly_flag, threshold


def _safe_auc(metric_fn, y_true: pd.Series, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metric_fn(y_true, y_score))


def defect_capture_rate(y_true: pd.Series, anomaly_score: np.ndarray, top_frac: float) -> float:
    n_total = len(y_true)
    n_top = max(1, int(np.ceil(n_total * top_frac)))
    ranked_idx = np.argsort(anomaly_score)
    top_idx = ranked_idx[:n_top]

    y_true_arr = np.asarray(y_true)
    captured_defects = int((y_true_arr[top_idx] == 1).sum())
    total_defects = int((y_true_arr == 1).sum())
    return safe_divide(captured_defects, total_defects)


def evaluate_predictions(y_true: pd.Series, anomaly_flag: np.ndarray, anomaly_score: np.ndarray) -> dict[str, object]:
    precision = float(precision_score(y_true, anomaly_flag, zero_division=0))
    recall = float(recall_score(y_true, anomaly_flag, zero_division=0))
    f1 = float(f1_score(y_true, anomaly_flag, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, anomaly_flag))

    roc_auc = _safe_auc(roc_auc_score, y_true, -anomaly_score)
    pr_auc = _safe_auc(average_precision_score, y_true, -anomaly_score)

    tn, fp, fn, tp = confusion_matrix(y_true, anomaly_flag, labels=[0, 1]).ravel()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "defect_capture_top_5pct": defect_capture_rate(y_true, anomaly_score, 0.05),
        "defect_capture_top_10pct": defect_capture_rate(y_true, anomaly_score, 0.10),
        "defect_capture_top_20pct": defect_capture_rate(y_true, anomaly_score, 0.20),
    }


def create_predictions_table(
    row_ids: pd.Series,
    y_true: pd.Series,
    anomaly_score: np.ndarray,
    anomaly_flag: np.ndarray,
    model_version: str = "isolation_forest_v1",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": row_ids.values,
            "quality_label_binary": y_true.values,
            "anomaly_score": anomaly_score,
            "anomaly_flag": anomaly_flag,
            "model_version": model_version,
            "run_ts": datetime.now(timezone.utc),
        }
    )


def compute_top_feature_shifts(clean_df: pd.DataFrame, feature_cols: list[str], anomaly_flag: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    local = clean_df.copy()
    local["anomaly_flag"] = anomaly_flag

    normal_df = local[local["anomaly_flag"] == 0]
    anomaly_df = local[local["anomaly_flag"] == 1]

    rows = []
    for col in feature_cols:
        normal_median = float(normal_df[col].median())
        anomaly_median = float(anomaly_df[col].median())
        normal_mad = float((normal_df[col] - normal_median).abs().median())
        robust_scale = normal_mad if normal_mad > 1e-6 else 1e-6
        shift = abs(anomaly_median - normal_median) / robust_scale
        rows.append(
            {
                "sensor": col,
                "normal_median": normal_median,
                "anomaly_median": anomaly_median,
                "robust_shift": shift,
            }
        )

    shifts = pd.DataFrame(rows).sort_values("robust_shift", ascending=False).head(top_n)
    return shifts


def explain_row_deviation(clean_df: pd.DataFrame, feature_cols: list[str], row_id: int, top_n: int = 5) -> pd.DataFrame:
    row = clean_df.loc[clean_df["row_id"] == row_id]
    if row.empty:
        return pd.DataFrame(columns=["sensor", "robust_z", "value", "median"])

    baseline = clean_df[feature_cols]
    medians = baseline.median()
    mads = (baseline - medians).abs().median().replace(0, 1e-6)

    deviations = []
    for col in feature_cols:
        value = float(row.iloc[0][col])
        med = float(medians[col])
        mad = float(mads[col])
        robust_z = abs(value - med) / mad
        deviations.append(
            {
                "sensor": col,
                "robust_z": robust_z,
                "value": value,
                "median": med,
            }
        )

    return pd.DataFrame(deviations).sort_values("robust_z", ascending=False).head(top_n)


def surrogate_feature_importance(x_df: pd.DataFrame, anomaly_flag: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    surrogate = RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE, class_weight="balanced")
    surrogate.fit(x_df, anomaly_flag)

    importances = pd.DataFrame(
        {
            "sensor": x_df.columns,
            "importance": surrogate.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importances.head(top_n)
