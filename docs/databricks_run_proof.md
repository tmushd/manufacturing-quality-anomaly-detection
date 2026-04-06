# Databricks Workspace Run Proof

## Workspace execution details
- Workspace host: `https://dbc-505eee04-18bb.cloud.databricks.com`
- Workspace user: `vyunndn@outlook.com`
- Notebook path: `/Workspace/Users/vyunndn@outlook.com/manufacturing-quality-anomaly-detection/proof_run_pipeline_embedded`
- Parent run ID: `430321928978342`
- Task run ID: `963645540221719`
- Run URL: https://dbc-505eee04-18bb.cloud.databricks.com/?o=7474660770902482#job/279557449134041/run/430321928978342

## Run state (from Jobs API)
```json
{
  "life_cycle_state": "TERMINATED",
  "result_state": "SUCCESS",
  "state_message": "",
  "user_cancelled_or_timedout": false
}
```

## Databricks SQL proof (workspace tables queried)
- SQL Warehouse ID: `c959f483dd118b17`
- Query timestamp (UTC): `2026-04-06T01:17:54.936504+00:00`

Query results:
- `bronze.secom_features_raw`: 1567
- `bronze.secom_labels_raw`: 1567
- `silver.secom_manufacturing_clean`: 1567
- `gold.secom_anomaly_predictions`: 1567
- `gold.secom_quality_summary`: `(bronze_rows=1567, silver_rows=1567, gold_rows=1567, raw_feature_count=591, final_feature_count=446, fail_count=104, dq_checks_total=7, dq_checks_passed=7)`
- `gold.data_quality_results`: `PASS=7`
- `gold.secom_quality_metrics`: `(accuracy=0.8851308232, precision=0.1545454545, recall=0.1634615385, f1=0.1588785047, balanced_accuracy=0.5508406799)`

## Tables created in workspace
- `bronze.secom_features_raw`
- `bronze.secom_labels_raw`
- `silver.secom_features_typed`
- `silver.secom_labels_typed`
- `silver.secom_manufacturing_clean`
- `gold.secom_feature_store`
- `gold.secom_anomaly_predictions`
- `gold.secom_quality_summary`
- `gold.data_quality_results`

## Raw proof artifacts in repo
- `outputs/reports/databricks_run_metadata.json`
- `outputs/reports/databricks_sql_proof.json`
