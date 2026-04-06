"""Alias notebook: transform Bronze tables into Silver outputs."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.transformations import build_silver_tables, write_silver_tables


def main() -> None:
    bronze_features = pd.read_parquet(config.BRONZE_DIR / "secom_features_raw.parquet")
    bronze_labels = pd.read_parquet(config.BRONZE_DIR / "secom_labels_raw.parquet")
    artifacts = build_silver_tables(bronze_features, bronze_labels)
    write_silver_tables(artifacts)
    print({"silver_rows": len(artifacts.manufacturing_clean)})


if __name__ == "__main__":
    main()
