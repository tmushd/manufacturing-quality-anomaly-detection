# Manufacturing Quality Anomaly Detection Pipeline (SECOM)

This project shows how I built a production-style semiconductor quality pipeline end to end using Databricks, PySpark, SQL, Delta Lake, and scikit-learn on the UCI SECOM dataset (1,567 records, 591 sensor features). I implemented a full Bronze/Silver/Gold medallion flow that ingests raw manufacturing signals, applies strict cleaning and typing rules, and publishes ML-ready Delta tables for downstream quality monitoring.

On top of the data platform layer, I trained an Isolation Forest anomaly model to flag likely defective units, added automated SQL data-quality controls, and produced interpretable outputs for high-risk records and sensor behavior shifts. The repository also includes verifiable Databricks workspace run proof (Jobs API + SQL query evidence), including an Isolation Forest accuracy of about 88.5% in the validated workspace run.

## Project Title
Manufacturing Quality Anomaly Detection Pipeline using Databricks, PySpark, SQL, Delta Lake, and Scikit-learn

## Dataset
- Source: UCI SECOM
- Records: 1,567
- Raw sensor features: 591
- Label mapping: `-1 -> 0 (pass)`, `1 -> 1 (fail)`
- Fail count in run: 104
- Files used:
  - `data/raw/secom.data`
  - `data/raw/secom_labels.data`

## Architecture
- Bronze: raw ingested records + ingestion metadata
- Silver: typed, cleaned, validated, joined records with explicit null policy
- Gold: ML-ready feature store, anomaly scores, quality summaries, and interpretability outputs

Local run persists tables as Parquet/CSV in `outputs/`; Databricks SQL notebooks/scripts are included for Delta table execution in workspace.

## Cleaning Rules
- Sensor values cast to numeric (`invalid -> null`)
- Drop columns with null ratio > 50%
- Drop constant/near-constant columns
- Median imputation for remaining numeric nulls
- Label standardization to binary pass/fail
- Deduplicate by `row_id`

## Run Results (Latest)
- Bronze rows: 1,567
- Silver curated rows: 1,567
- Gold feature rows: 1,567
- Features after cleaning: 440
- Dropped high-null features: 29
- Dropped constant/near-constant features: 122
- Data quality checks: 8/8 PASS

### Model Metrics (Isolation Forest)
- Precision: 0.1636
- Recall: 0.1731
- F1: 0.1682
- Balanced Accuracy: 0.5551
- ROC-AUC: 0.5896
- PR-AUC: 0.1161
- Defect capture @ top 10% anomalies: 0.1923

Confusion matrix:
- TN: 1371
- FP: 92
- FN: 86
- TP: 18

## SQL Validation Checks Implemented
1. Row-count reconciliation
2. Schema integrity
3. Required field null checks
4. Label integrity (`-1/1` raw, `0/1` standardized)
5. Duplicate detection by `row_id`
6. Cast-failure tracking
7. Null-threshold compliance
8. Gold scoring completeness

## Interpretability Outputs
- Global: top 10 anomaly-associated sensors using robust anomaly-vs-normal shift
- Surrogate: top 10 features from RandomForest surrogate trained on anomaly flags
- Local: top 5 deviating sensors for 3 flagged records

## Repository Structure
```text
manufacturing-quality-anomaly-detection/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_setup_and_ingestion.py
│   ├── 02_bronze_to_silver_transform.py
│   ├── 03_sql_data_quality_checks.sql
│   ├── 04_build_gold_feature_store.py
│   ├── 05_train_isolation_forest.py
│   ├── 06_evaluate_and_explain.py
│   └── 07_dashboard_queries.sql
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── transformations.py
│   ├── quality_checks.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── pipeline.py
├── sql/
│   ├── create_catalog_schemas.sql
│   ├── validation_checks.sql
│   └── dashboard_queries.sql
├── tests/
│   ├── conftest.py
│   ├── test_schema.py
│   ├── test_transformations.py
│   └── test_model.py
├── docs/
│   ├── architecture_diagram.png
│   ├── dashboard_screenshot.png
│   ├── data_dictionary.md
│   └── final_report.md
└── outputs/
    ├── bronze/
    ├── silver/
    ├── gold/
    ├── reports/
    └── figures/
```

## Run Order
1. `python3 notebooks/01_setup_and_ingestion.py`
2. `python3 notebooks/02_bronze_to_silver_transform.py`
3. Execute `notebooks/03_sql_data_quality_checks.sql` in Databricks SQL
4. `python3 notebooks/04_build_gold_feature_store.py`
5. `python3 notebooks/05_train_isolation_forest.py`
6. `python3 notebooks/06_evaluate_and_explain.py`
7. Execute `notebooks/07_dashboard_queries.sql` in Databricks SQL

Or run everything in one shot:
- `python3 -m src.pipeline`

## Key Artifacts
- `outputs/gold/secom_feature_store.parquet`
- `outputs/gold/secom_anomaly_predictions.parquet`
- `outputs/gold/secom_quality_summary.parquet`
- `outputs/gold/data_quality_results.csv`
- `outputs/reports/model_metrics.json`
- `outputs/reports/top_feature_shifts.csv`
- `outputs/gold/local_row_explanations.csv`
- `docs/architecture_diagram.png`
- `docs/dashboard_screenshot.png`

## Summary
Built a Databricks-style ELT pipeline with PySpark/SQL-ready assets to ingest and transform 1,567 semiconductor records with 591 raw features into Bronze/Silver/Gold layers, trained an Isolation Forest for defect-oriented anomaly detection, implemented 8 automated SQL quality checks, and reduced invalid curated records to zero under defined acceptance rules.


## Databricks Workspace
- Jobs run: `docs/databricks_run_proof.md`
- Raw Jobs API metadata: `outputs/reports/databricks_run_metadata.json`
- Raw SQL validation proof: `outputs/reports/databricks_sql_proof.json`
- Isolation Forest accuracy (Databricks run): `0.8851308232291002` (~`88.51%`)
