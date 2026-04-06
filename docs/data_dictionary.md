# Data Dictionary

## Bronze
### `bronze.secom_features_raw`
- `row_id` (BIGINT): stable row index
- `sensor_001` ... `sensor_591` (STRING in raw ingest)
- `source_file` (STRING)
- `ingest_ts` (TIMESTAMP)
- `batch_id` (STRING)

### `bronze.secom_labels_raw`
- `row_id` (BIGINT)
- `raw_label` (STRING)
- `raw_timestamp` (STRING)
- `source_file` (STRING)
- `ingest_ts` (TIMESTAMP)
- `batch_id` (STRING)

## Silver
### `silver.secom_features_typed`
- `row_id` (BIGINT)
- remaining `sensor_*` columns after high-null/constant filtering (DOUBLE)
- ingest metadata

### `silver.secom_labels_typed`
- `row_id` (BIGINT)
- `quality_label_raw` (INT, expected `-1` or `1`)
- `quality_label_binary` (INT, `0` pass / `1` fail)
- `event_ts` (TIMESTAMP)

### `silver.secom_manufacturing_clean`
- `row_id` (BIGINT)
- cleaned numeric feature columns
- `quality_label_raw` (INT)
- `quality_label_binary` (INT)
- `event_ts` (TIMESTAMP)

## Gold
### `gold.secom_feature_store`
- `row_id` (BIGINT)
- `quality_label_binary` (INT)
- final model feature columns (DOUBLE)

### `gold.secom_anomaly_predictions`
- `row_id` (BIGINT)
- `quality_label_binary` (INT)
- `anomaly_score` (DOUBLE, lower = more anomalous)
- `anomaly_flag` (INT, 1 = anomalous)
- `model_version` (STRING)
- `run_ts` (TIMESTAMP)

### `gold.secom_quality_summary`
- run-level summary metrics and dataset stats

### `gold.data_quality_results`
- `check_name` (STRING)
- `check_status` (STRING)
- `failed_count` (INT)
- `details` (STRING)
- `check_ts` (TIMESTAMP)
