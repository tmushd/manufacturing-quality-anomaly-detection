from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src import config
from src.utils import get_logger


logger = get_logger(__name__)


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("File already exists, skipping download: %s", destination)
        return destination

    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    logger.info("Saved file to %s", destination)
    return destination


def download_secom_dataset() -> tuple[Path, Path]:
    config.ensure_project_dirs()
    features_path = download_file(config.FEATURES_URL, config.FEATURES_FILE)
    labels_path = download_file(config.LABELS_URL, config.LABELS_FILE)
    download_file(config.NAMES_URL, config.NAMES_FILE)
    return features_path, labels_path


def _build_sensor_columns(count: int) -> list[str]:
    return [f"sensor_{idx:03d}" for idx in range(1, count + 1)]


def load_features_raw(path: Path) -> pd.DataFrame:
    sensor_cols = _build_sensor_columns(config.EXPECTED_FEATURES)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=sensor_cols,
        na_values=["NaN", "nan", "NAN", "?"],
        engine="python",
    )
    ingest_ts = datetime.now(timezone.utc)
    df.insert(0, "row_id", range(1, len(df) + 1))
    df["source_file"] = path.name
    df["ingest_ts"] = ingest_ts
    df["batch_id"] = ingest_ts.strftime("%Y%m%d%H%M%S")
    return df


def load_labels_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["raw_label", "raw_date", "raw_time"], engine="python")
    raw["raw_timestamp"] = (raw["raw_date"].astype(str).str.strip('"') + " " + raw["raw_time"].astype(str).str.strip('"')).str.strip()
    raw = raw.drop(columns=["raw_date", "raw_time"])
    ingest_ts = datetime.now(timezone.utc)
    raw.insert(0, "row_id", range(1, len(raw) + 1))
    raw["source_file"] = path.name
    raw["ingest_ts"] = ingest_ts
    raw["batch_id"] = ingest_ts.strftime("%Y%m%d%H%M%S")
    return raw


def write_bronze_tables(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> dict[str, Path]:
    config.BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "secom_features_raw_parquet": config.BRONZE_DIR / "secom_features_raw.parquet",
        "secom_labels_raw_parquet": config.BRONZE_DIR / "secom_labels_raw.parquet",
        "secom_features_raw_csv": config.BRONZE_DIR / "secom_features_raw.csv",
        "secom_labels_raw_csv": config.BRONZE_DIR / "secom_labels_raw.csv",
    }

    features_df.to_parquet(paths["secom_features_raw_parquet"], index=False)
    labels_df.to_parquet(paths["secom_labels_raw_parquet"], index=False)
    features_df.to_csv(paths["secom_features_raw_csv"], index=False)
    labels_df.to_csv(paths["secom_labels_raw_csv"], index=False)

    logger.info("Wrote bronze tables to %s", config.BRONZE_DIR)
    return paths
