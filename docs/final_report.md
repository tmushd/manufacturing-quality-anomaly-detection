# Final Project Report

## Objective
Build a production-style medallion pipeline for semiconductor quality anomaly detection using SECOM, with automated quality checks and interpretable outputs.

## Dataset Snapshot
- Records: 1,567
- Raw features: 591
- Label encoding (raw): `-1` pass, `1` fail
- Fail records: 104

## Pipeline Summary
- Bronze: raw feature + label ingestion with metadata
- Silver: typed, null-managed, imputed, deduplicated, label-standardized joined table
- Gold: feature store, prediction table, quality summary, anomaly explainability outputs

## Data Cleaning Outcome
- High-null feature drop threshold: >50%
- High-null columns dropped: 29
- Constant/near-constant columns dropped: 122
- Final model feature count: 440

## Data Quality Validation Outcome
All 8 checks passed:
1. Row-count reconciliation
2. Schema integrity
3. Required nulls
4. Label integrity
5. Duplicate detection
6. Cast-failure tracking
7. Null-threshold compliance
8. Gold scoring completeness

## Model Configuration
Isolation Forest parameters:
- `n_estimators=200`
- `max_samples='auto'`
- `max_features=0.6`
- `contamination=0.07`
- `random_state=42`

Training mode:
- trained on pass class only
- scored all records

## Evaluation
Full-dataset results:
- Precision: 0.1636
- Recall: 0.1731
- F1: 0.1682
- Balanced Accuracy: 0.5551
- ROC-AUC: 0.5896
- PR-AUC: 0.1161

Confusion matrix:
- TN=1371, FP=92, FN=86, TP=18

Defect capture:
- Top 5% anomalies: 13.46%
- Top 10% anomalies: 19.23%
- Top 20% anomalies: 30.77%

## Interpretability
Global:
- top 10 sensors by robust anomaly-vs-normal shift
- surrogate feature importance (RandomForest on anomaly flags)

Local:
- top 5 deviating sensors for 3 anomalous records in `outputs/gold/local_row_explanations.csv`

## Portfolio Deliverables Completed
- Bronze/Silver/Gold artifacts generated
- SQL checks authored and executed in pipeline
- Isolation Forest trained and persisted outputs created
- Evaluation + interpretability reports generated
- Architecture and dashboard visuals generated
- README and data dictionary completed
