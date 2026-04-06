-- Notebook 03: SQL Data Quality Checks (Databricks SQL)
-- Replace table references with your catalog/schema qualifiers in Databricks.

-- 1) Row-count reconciliation
SELECT
  (SELECT COUNT(*) FROM bronze.secom_features_raw) AS feature_rows,
  (SELECT COUNT(*) FROM bronze.secom_labels_raw) AS label_rows;

-- 2) Schema integrity
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'silver'
  AND table_name = 'secom_manufacturing_clean'
ORDER BY ordinal_position;

-- 3) Required field null checks
SELECT COUNT(*) AS invalid_required_nulls
FROM silver.secom_manufacturing_clean
WHERE row_id IS NULL OR quality_label_binary IS NULL;

-- 4) Label integrity checks
SELECT quality_label_raw, COUNT(*) AS cnt
FROM silver.secom_labels_typed
GROUP BY quality_label_raw;

SELECT quality_label_binary, COUNT(*) AS cnt
FROM silver.secom_labels_typed
GROUP BY quality_label_binary;

-- 5) Duplicate detection
SELECT row_id, COUNT(*) AS dup_count
FROM silver.secom_manufacturing_clean
GROUP BY row_id
HAVING COUNT(*) > 1;

-- 6) Cast-failure / invalid numeric tracking
SELECT check_name, check_status, failed_count, details, check_ts
FROM gold.data_quality_results
WHERE check_name = 'cast_failure_tracking'
ORDER BY check_ts DESC;

-- 7) Feature null-threshold compliance
SELECT check_name, check_status, failed_count, details, check_ts
FROM gold.data_quality_results
WHERE check_name = 'null_threshold_compliance'
ORDER BY check_ts DESC;

-- 8) Gold scoring completeness
SELECT COUNT(*) AS missing_scores
FROM gold.secom_anomaly_predictions
WHERE anomaly_score IS NULL OR anomaly_flag IS NULL;
