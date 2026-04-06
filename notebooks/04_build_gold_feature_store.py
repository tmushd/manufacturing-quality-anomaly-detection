"""Notebook 04: Build Gold feature store."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.feature_engineering import select_model_features
from src.utils import write_json


def main() -> None:
    clean_df = pd.read_parquet(config.SILVER_DIR / "secom_manufacturing_clean.parquet")
    feature_cols = select_model_features(clean_df)

    feature_store = clean_df[["row_id", "quality_label_binary", *feature_cols]].copy()
    feature_store.to_parquet(config.GOLD_DIR / "secom_feature_store.parquet", index=False)
    feature_store.to_csv(config.GOLD_DIR / "secom_feature_store.csv", index=False)

    summary = {
        "gold_feature_rows": len(feature_store),
        "gold_feature_count": len(feature_cols),
        "null_cells": int(feature_store[feature_cols].isna().sum().sum()),
    }
    write_json(summary, config.REPORTS_DIR / "notebook_04_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
