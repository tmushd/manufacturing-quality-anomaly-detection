from src import config


def test_expected_dataset_shape_constants() -> None:
    assert config.EXPECTED_ROWS == 1567
    assert config.EXPECTED_FEATURES == 591
    assert 0 < config.NULL_DROP_THRESHOLD < 1
