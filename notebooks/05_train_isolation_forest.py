"""Notebook 05: Train Isolation Forest and persist predictions."""

from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.feature_engineering import prepare_training_matrix
from src.modeling import create_predictions_table, score_records, split_data, train_isolation_forest
from src.utils import write_json


def main() -> None:
    clean_df = pd.read_parquet(config.SILVER_DIR / "secom_manufacturing_clean.parquet")
    x_df, y, _ = prepare_training_matrix(clean_df)
    x_train, _, y_train, _ = split_data(x_df, y)

    model = train_isolation_forest(x_train, y_train, train_on_pass_only=True)
    scores, flags, threshold = score_records(model, x_df)

    predictions = create_predictions_table(clean_df["row_id"], y, scores, flags)
    predictions.to_parquet(config.GOLD_DIR / "secom_anomaly_predictions.parquet", index=False)
    predictions.to_csv(config.GOLD_DIR / "secom_anomaly_predictions.csv", index=False)

    model_path = config.GOLD_DIR / "isolation_forest_model.joblib"
    joblib.dump(model, model_path)

    summary = {
        "predictions_rows": len(predictions),
        "anomaly_count": int(predictions["anomaly_flag"].sum()),
        "threshold": threshold,
        "model_path": str(model_path),
    }
    write_json(summary, config.REPORTS_DIR / "notebook_05_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
