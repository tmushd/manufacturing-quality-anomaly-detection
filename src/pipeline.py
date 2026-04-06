from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import prepare_training_matrix
from src.ingestion import download_secom_dataset, load_features_raw, load_labels_raw, write_bronze_tables
from src.modeling import (
    compute_top_feature_shifts,
    create_predictions_table,
    evaluate_predictions,
    explain_row_deviation,
    score_records,
    split_data,
    surrogate_feature_importance,
    train_isolation_forest,
)
from src.quality_checks import persist_check_results, run_all_checks
from src.transformations import build_silver_tables, write_silver_tables
from src.utils import get_logger, write_json


logger = get_logger(__name__)


def _write_gold_tables(
    feature_store_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    quality_summary_df: pd.DataFrame,
    top_anomalies_df: pd.DataFrame,
    global_shift_df: pd.DataFrame,
    surrogate_df: pd.DataFrame,
    local_explanations_df: pd.DataFrame,
) -> None:
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)

    feature_store_df.to_parquet(config.GOLD_DIR / "secom_feature_store.parquet", index=False)
    feature_store_df.to_csv(config.GOLD_DIR / "secom_feature_store.csv", index=False)

    predictions_df.to_parquet(config.GOLD_DIR / "secom_anomaly_predictions.parquet", index=False)
    predictions_df.to_csv(config.GOLD_DIR / "secom_anomaly_predictions.csv", index=False)

    quality_summary_df.to_parquet(config.GOLD_DIR / "secom_quality_summary.parquet", index=False)
    quality_summary_df.to_csv(config.GOLD_DIR / "secom_quality_summary.csv", index=False)

    top_anomalies_df.to_parquet(config.GOLD_DIR / "top_abnormal_records.parquet", index=False)
    top_anomalies_df.to_csv(config.GOLD_DIR / "top_abnormal_records.csv", index=False)

    global_shift_df.to_parquet(config.GOLD_DIR / "top_feature_shift_analysis.parquet", index=False)
    global_shift_df.to_csv(config.GOLD_DIR / "top_feature_shift_analysis.csv", index=False)

    surrogate_df.to_parquet(config.GOLD_DIR / "surrogate_feature_importance.parquet", index=False)
    surrogate_df.to_csv(config.GOLD_DIR / "surrogate_feature_importance.csv", index=False)

    local_explanations_df.to_parquet(config.GOLD_DIR / "local_row_explanations.parquet", index=False)
    local_explanations_df.to_csv(config.GOLD_DIR / "local_row_explanations.csv", index=False)


def _plot_anomaly_distribution(predictions_df: pd.DataFrame) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = config.FIGURES_DIR / "anomaly_score_distribution.png"

    pass_scores = predictions_df.loc[predictions_df["quality_label_binary"] == 0, "anomaly_score"]
    fail_scores = predictions_df.loc[predictions_df["quality_label_binary"] == 1, "anomaly_score"]

    plt.figure(figsize=(10, 6))
    plt.hist(pass_scores, bins=40, alpha=0.7, label="Pass (0)")
    plt.hist(fail_scores, bins=40, alpha=0.7, label="Fail (1)")
    plt.title("Isolation Forest Anomaly Score Distribution")
    plt.xlabel("Anomaly Score (Lower = More Anomalous)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()
    return fig_path


def _plot_confusion_matrix(metrics: dict[str, object]) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = config.FIGURES_DIR / "confusion_matrix.png"
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Pass (0)", "Fail (1)"])
    plt.yticks([0, 1], ["Pass (0)", "Fail (1)"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()
    return fig_path


def _build_architecture_diagram() -> Path:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = config.DOCS_DIR / "architecture_diagram.png"

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.axis("off")

    boxes = [
        (0.05, 0.3, 0.22, 0.4, "Bronze\nRaw SECOM Files\nDelta Tables"),
        (0.39, 0.3, 0.22, 0.4, "Silver\nTyped + Cleaned + Imputed\nValidated Dataset"),
        (0.73, 0.3, 0.22, 0.4, "Gold\nFeature Store + Scores\nQuality Summaries"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#d9edf7", edgecolor="#31708f", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(0.39, 0.5), xytext=(0.27, 0.5), arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.annotate("", xy=(0.73, 0.5), xytext=(0.61, 0.5), arrowprops=dict(arrowstyle="->", linewidth=2))

    ax.text(0.5, 0.08, "Databricks + PySpark + SQL + Delta Lake + Scikit-learn", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return fig_path


def _build_dashboard_snapshot(metrics: dict[str, object], top_anomalies_df: pd.DataFrame, top_shifts_df: pd.DataFrame) -> Path:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DOCS_DIR / "dashboard_screenshot.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SECOM Quality Monitoring Snapshot", fontsize=14)

    metric_rows = [
        ["Precision", f"{metrics['precision']:.3f}"],
        ["Recall", f"{metrics['recall']:.3f}"],
        ["F1", f"{metrics['f1']:.3f}"],
        ["Balanced Acc", f"{metrics['balanced_accuracy']:.3f}"],
        ["ROC-AUC", f"{metrics['roc_auc']:.3f}"],
        ["PR-AUC", f"{metrics['pr_auc']:.3f}"],
    ]
    axes[0].axis("off")
    table = axes[0].table(cellText=metric_rows, colLabels=["Metric", "Value"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    axes[0].set_title("Model Metrics")

    top_rows = top_anomalies_df.head(8)[["row_id", "quality_label_binary", "anomaly_score"]]
    axes[1].axis("off")
    table2 = axes[1].table(cellText=top_rows.values, colLabels=top_rows.columns, loc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.1, 1.2)
    axes[1].set_title("Top Anomalous Records")

    shift_rows = top_shifts_df.head(8)[["sensor", "robust_shift"]]
    axes[2].axis("off")
    table3 = axes[2].table(cellText=shift_rows.values, colLabels=shift_rows.columns, loc="center")
    table3.auto_set_font_size(False)
    table3.set_fontsize(8)
    table3.scale(1.1, 1.2)
    axes[2].set_title("Top Sensor Shifts")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_full_pipeline() -> dict[str, object]:
    logger.info("Starting full SECOM pipeline run")
    config.ensure_project_dirs()

    features_path, labels_path = download_secom_dataset()
    bronze_features = load_features_raw(features_path)
    bronze_labels = load_labels_raw(labels_path)
    write_bronze_tables(bronze_features, bronze_labels)

    silver_artifacts = build_silver_tables(bronze_features, bronze_labels)
    write_silver_tables(silver_artifacts)

    clean_df = silver_artifacts.manufacturing_clean.copy()
    x_df, y, feature_cols = prepare_training_matrix(clean_df)
    x_train, x_test, y_train, y_test = split_data(x_df, y)

    model = train_isolation_forest(x_train, y_train, train_on_pass_only=True)

    full_scores, full_flags, threshold = score_records(model, x_df)
    full_predictions = create_predictions_table(clean_df["row_id"], y, full_scores, full_flags)

    test_scores, test_flags, _ = score_records(model, x_test)
    test_metrics = evaluate_predictions(y_test, test_flags, test_scores)
    full_metrics = evaluate_predictions(y, full_flags, full_scores)

    feature_store = clean_df[["row_id", "quality_label_binary", *feature_cols]].copy()

    top_anomalies = (
        full_predictions.sort_values("anomaly_score", ascending=True)
        .head(20)
        .merge(clean_df[["row_id", "quality_label_binary", "event_ts"]], on=["row_id", "quality_label_binary"], how="left")
    )

    global_shift = compute_top_feature_shifts(clean_df, feature_cols, full_flags, top_n=10)
    surrogate_top = surrogate_feature_importance(x_df, full_flags, top_n=10)

    flagged_rows = full_predictions.loc[full_predictions["anomaly_flag"] == 1, "row_id"].head(3).tolist()
    local_parts = []
    for rid in flagged_rows:
        local = explain_row_deviation(clean_df, feature_cols, int(rid), top_n=5)
        local.insert(0, "row_id", int(rid))
        local_parts.append(local)
    local_explanations = pd.concat(local_parts, ignore_index=True) if local_parts else pd.DataFrame()

    quality_summary = pd.DataFrame(
        [
            {
                "run_ts": datetime.now(timezone.utc),
                "dataset_rows": len(clean_df),
                "feature_count_after_cleaning": len(feature_cols),
                "fails": int((clean_df["quality_label_binary"] == 1).sum()),
                "passes": int((clean_df["quality_label_binary"] == 0).sum()),
                "threshold": threshold,
                "precision": full_metrics["precision"],
                "recall": full_metrics["recall"],
                "f1": full_metrics["f1"],
                "balanced_accuracy": full_metrics["balanced_accuracy"],
                "roc_auc": full_metrics["roc_auc"],
                "pr_auc": full_metrics["pr_auc"],
            }
        ]
    )

    _write_gold_tables(
        feature_store_df=feature_store,
        predictions_df=full_predictions,
        quality_summary_df=quality_summary,
        top_anomalies_df=top_anomalies,
        global_shift_df=global_shift,
        surrogate_df=surrogate_top,
        local_explanations_df=local_explanations,
    )

    dq_results = run_all_checks(
        bronze_features_df=bronze_features,
        bronze_labels_df=bronze_labels,
        silver_features_typed_df=silver_artifacts.features_typed,
        silver_labels_typed_df=silver_artifacts.labels_typed,
        silver_clean_df=silver_artifacts.manufacturing_clean,
        gold_predictions_df=full_predictions,
    )
    persist_check_results(dq_results)

    metrics_report = {
        "full_dataset_metrics": full_metrics,
        "test_dataset_metrics": test_metrics,
        "model_threshold": threshold,
        "dropped_high_null_columns": silver_artifacts.dropped_high_null_cols,
        "dropped_constant_columns": silver_artifacts.dropped_constant_cols,
        "feature_count_after_cleaning": len(feature_cols),
    }
    write_json(metrics_report, config.REPORTS_DIR / "model_metrics.json")
    dq_results.to_csv(config.REPORTS_DIR / "data_quality_results.csv", index=False)
    global_shift.to_csv(config.REPORTS_DIR / "top_feature_shifts.csv", index=False)
    surrogate_top.to_csv(config.REPORTS_DIR / "surrogate_feature_importance.csv", index=False)
    local_explanations.to_csv(config.REPORTS_DIR / "local_row_explanations.csv", index=False)

    anomaly_plot = _plot_anomaly_distribution(full_predictions)
    confusion_plot = _plot_confusion_matrix(full_metrics)
    arch_plot = _build_architecture_diagram()
    dashboard_plot = _build_dashboard_snapshot(full_metrics, top_anomalies, global_shift)

    outputs = {
        "bronze_rows": len(bronze_features),
        "silver_rows": len(silver_artifacts.manufacturing_clean),
        "gold_rows": len(feature_store),
        "feature_count_after_cleaning": len(feature_cols),
        "fail_count": int((clean_df["quality_label_binary"] == 1).sum()),
        "dq_passed": int((dq_results["check_status"] == "PASS").sum()),
        "dq_total": len(dq_results),
        "figures": {
            "anomaly_distribution": str(anomaly_plot),
            "confusion_matrix": str(confusion_plot),
            "architecture_diagram": str(arch_plot),
            "dashboard_snapshot": str(dashboard_plot),
        },
    }

    write_json(outputs, config.REPORTS_DIR / "run_summary.json")
    logger.info("Pipeline run complete")
    return outputs


if __name__ == "__main__":
    result = run_full_pipeline()
    print(result)
