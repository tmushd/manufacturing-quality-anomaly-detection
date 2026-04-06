"""Alias notebook: ingest raw SECOM files into Bronze outputs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import download_secom_dataset, load_features_raw, load_labels_raw, write_bronze_tables


def main() -> None:
    features_path, labels_path = download_secom_dataset()
    features_df = load_features_raw(features_path)
    labels_df = load_labels_raw(labels_path)
    write_bronze_tables(features_df, labels_df)
    print({"bronze_features_rows": len(features_df), "bronze_labels_rows": len(labels_df)})


if __name__ == "__main__":
    main()
