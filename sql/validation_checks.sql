-- Automated validation checks

-- 1) Row-count reconciliation
SELECT
  (SELECT COUNT(*) FROM bronze.secom_features_raw) AS feature_rows,
  (SELECT COUNT(*) FROM bronze.secom_labels_raw) AS label_rows,
  CASE WHEN (SELECT COUNT(*) FROM bronze.secom_features_raw) = (SELECT COUNT(*) FROM bronze.secom_labels_raw)
    THEN 'PASS' ELSE 'FAIL' END AS status;

-- 2) Required-null check
SELECT COUNT(*) AS invalid_required_nulls
FROM silver.secom_manufacturing_clean
WHERE row_id IS NULL OR quality_label_binary IS NULL;

-- 3) Label integrity
SELECT quality_label_binary, COUNT(*)
FROM silver.secom_labels_typed
GROUP BY quality_label_binary;

-- 4) Duplicate detection
SELECT row_id, COUNT(*) AS dup_count
FROM silver.secom_manufacturing_clean
GROUP BY row_id
HAVING COUNT(*) > 1;

-- 5) Gold completeness
SELECT COUNT(*) AS incomplete_scoring_rows
FROM gold.secom_anomaly_predictions
WHERE anomaly_score IS NULL OR anomaly_flag IS NULL;

-- 6) Latest quality result dashboard
SELECT *
FROM gold.data_quality_results
ORDER BY check_ts DESC;
