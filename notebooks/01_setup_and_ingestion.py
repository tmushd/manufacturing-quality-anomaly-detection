"""Notebook 01: Setup and Bronze ingestion."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.ingestion import download_secom_dataset, load_features_raw, load_labels_raw, write_bronze_tables
from src.utils import write_json


def main() -> None:
    config.ensure_project_dirs()
    features_path, labels_path = download_secom_dataset()

    bronze_features = load_features_raw(features_path)
    bronze_labels = load_labels_raw(labels_path)
    write_bronze_tables(bronze_features, bronze_labels)

    summary = {
        "bronze_features_rows": len(bronze_features),
        "bronze_labels_rows": len(bronze_labels),
        "bronze_feature_columns": len([c for c in bronze_features.columns if c.startswith('sensor_')]),
    }
    write_json(summary, config.REPORTS_DIR / "notebook_01_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
