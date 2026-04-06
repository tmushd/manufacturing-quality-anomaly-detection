import pandas as pd

from src.transformations import (
    cast_sensor_columns,
    drop_constant_and_near_constant_columns,
    drop_high_null_columns,
    impute_median,
)


def test_drop_high_null_columns() -> None:
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "sensor_001": [1.0, None, None, None],
            "sensor_002": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out, dropped, _ = drop_high_null_columns(df, ["sensor_001", "sensor_002"], threshold=0.5)
    assert "sensor_001" in dropped
    assert "sensor_001" not in out.columns


def test_impute_median_fills_nulls() -> None:
    df = pd.DataFrame({"sensor_001": [1.0, None, 3.0], "sensor_002": [2.0, 2.0, None]})
    imputed, _ = impute_median(df, ["sensor_001", "sensor_002"])
    assert imputed.isna().sum().sum() == 0


def test_drop_constant_columns() -> None:
    df = pd.DataFrame({"sensor_001": [7, 7, 7], "sensor_002": [1, 2, 3]})
    out, dropped = drop_constant_and_near_constant_columns(df, ["sensor_001", "sensor_002"], 0.99)
    assert "sensor_001" in dropped
    assert "sensor_002" in out.columns


def test_cast_sensor_columns() -> None:
    df = pd.DataFrame({"sensor_001": ["1.0", "bad"], "sensor_002": ["2", "3"]})
    out = cast_sensor_columns(df)
    assert out["sensor_001"].isna().sum() == 1
    assert str(out["sensor_002"].dtype).startswith(("int", "float"))
