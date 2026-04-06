from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = ROOT_DIR / "outputs"
BRONZE_DIR = OUTPUTS_DIR / "bronze"
SILVER_DIR = OUTPUTS_DIR / "silver"
GOLD_DIR = OUTPUTS_DIR / "gold"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"

FEATURES_FILE = RAW_DIR / "secom.data"
LABELS_FILE = RAW_DIR / "secom_labels.data"
NAMES_FILE = RAW_DIR / "secom.names"

FEATURES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.names"

EXPECTED_ROWS = 1567
EXPECTED_FEATURES = 591
NULL_DROP_THRESHOLD = 0.50
NEAR_CONSTANT_THRESHOLD = 0.995
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 200
    max_samples: str = "auto"
    max_features: float = 0.6
    contamination: float = 0.07
    random_state: int = RANDOM_STATE


MODEL_CONFIG = ModelConfig()


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DIR,
        OUTPUTS_DIR,
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
        DOCS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
