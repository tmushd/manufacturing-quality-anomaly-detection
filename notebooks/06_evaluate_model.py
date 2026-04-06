"""Notebook 06: Evaluate model and build interpretability outputs."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.feature_engineering import prepare_training_matrix
from src.modeling import compute_top_feature_shifts, evaluate_predictions, explain_row_deviation, surrogate_feature_importance
from src.utils import write_json


def main() -> None:
    clean_df = pd.read_parquet(config.SILVER_DIR / "secom_manufacturing_clean.parquet")
    predictions = pd.read_parquet(config.GOLD_DIR / "secom_anomaly_predictions.parquet")

    x_df, y, feature_cols = prepare_training_matrix(clean_df)
    scores = predictions["anomaly_score"].to_numpy()
    flags = predictions["anomaly_flag"].to_numpy()

    metrics = evaluate_predictions(y, flags, scores)
    top_shifts = compute_top_feature_shifts(clean_df, feature_cols, flags, top_n=10)
    surrogate = surrogate_feature_importance(x_df, flags, top_n=10)

    flagged_ids = predictions.loc[predictions["anomaly_flag"] == 1, "row_id"].head(3).tolist()
    local_parts = []
    for rid in flagged_ids:
        local = explain_row_deviation(clean_df, feature_cols, int(rid), top_n=5)
        local.insert(0, "row_id", int(rid))
        local_parts.append(local)
    local_explanations = pd.concat(local_parts, ignore_index=True) if local_parts else pd.DataFrame()

    top_shifts.to_csv(config.REPORTS_DIR / "top_feature_shifts.csv", index=False)
    surrogate.to_csv(config.REPORTS_DIR / "surrogate_feature_importance.csv", index=False)
    local_explanations.to_csv(config.REPORTS_DIR / "local_row_explanations.csv", index=False)
    write_json(metrics, config.REPORTS_DIR / "evaluation_metrics.json")

    print(metrics)


if __name__ == "__main__":
    main()
