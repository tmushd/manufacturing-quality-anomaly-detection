"""Notebook 02: Bronze to Silver transformations."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.transformations import build_silver_tables, write_silver_tables
from src.utils import write_json


def main() -> None:
    bronze_features = pd.read_parquet(config.BRONZE_DIR / "secom_features_raw.parquet")
    bronze_labels = pd.read_parquet(config.BRONZE_DIR / "secom_labels_raw.parquet")

    artifacts = build_silver_tables(bronze_features, bronze_labels)
    write_silver_tables(artifacts)

    summary = {
        "silver_rows": len(artifacts.manufacturing_clean),
        "dropped_high_null_cols": len(artifacts.dropped_high_null_cols),
        "dropped_constant_cols": len(artifacts.dropped_constant_cols),
        "remaining_sensor_features": len([c for c in artifacts.features_typed.columns if c.startswith('sensor_')]),
    }
    write_json(summary, config.REPORTS_DIR / "notebook_02_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
