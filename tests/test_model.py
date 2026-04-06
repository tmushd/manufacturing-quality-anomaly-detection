import numpy as np
import pandas as pd

from src.modeling import defect_capture_rate, evaluate_predictions


def test_defect_capture_rate_bounds() -> None:
    y = pd.Series([0, 1, 0, 1, 0, 0])
    scores = np.array([-0.9, -0.8, -0.2, -0.7, -0.1, -0.05])
    result = defect_capture_rate(y, scores, 0.5)
    assert 0.0 <= result <= 1.0


def test_evaluate_predictions_keys() -> None:
    y = pd.Series([0, 0, 1, 1])
    flags = np.array([0, 1, 1, 1])
    scores = np.array([-0.1, -0.8, -0.9, -0.7])
    metrics = evaluate_predictions(y, flags, scores)
    for key in ["precision", "recall", "f1", "balanced_accuracy", "confusion_matrix"]:
        assert key in metrics
