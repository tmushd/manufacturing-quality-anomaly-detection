-- Notebook 07: Dashboard SQL queries (Databricks SQL)

-- A) Pass vs fail counts
SELECT quality_label_binary, COUNT(*) AS record_count
FROM silver.secom_manufacturing_clean
GROUP BY quality_label_binary
ORDER BY quality_label_binary;

-- B) Predicted anomaly vs true label
SELECT quality_label_binary, anomaly_flag, COUNT(*) AS cnt
FROM gold.secom_anomaly_predictions
GROUP BY quality_label_binary, anomaly_flag
ORDER BY quality_label_binary, anomaly_flag;

-- C) Top 20 highest-risk records
SELECT row_id, quality_label_binary, anomaly_score
FROM gold.secom_anomaly_predictions
ORDER BY anomaly_score ASC
LIMIT 20;

-- D) Top-N defect capture rates
WITH ranked AS (
  SELECT
    row_id,
    quality_label_binary,
    anomaly_score,
    ROW_NUMBER() OVER (ORDER BY anomaly_score ASC) AS rn,
    COUNT(*) OVER () AS total_rows
  FROM gold.secom_anomaly_predictions
),
thresholded AS (
  SELECT
    quality_label_binary,
    CASE
      WHEN rn <= total_rows * 0.05 THEN 'top_5pct'
      WHEN rn <= total_rows * 0.10 THEN 'top_10pct'
      WHEN rn <= total_rows * 0.20 THEN 'top_20pct'
      ELSE 'rest'
    END AS bucket
  FROM ranked
)
SELECT bucket, SUM(CASE WHEN quality_label_binary = 1 THEN 1 ELSE 0 END) AS fail_count
FROM thresholded
GROUP BY bucket
ORDER BY bucket;

-- E) Data quality check results
SELECT check_name, check_status, failed_count, details, check_ts
FROM gold.data_quality_results
ORDER BY check_ts DESC;
