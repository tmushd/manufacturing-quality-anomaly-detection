-- Dashboard queries

SELECT quality_label_binary, COUNT(*) AS record_count
FROM silver.secom_manufacturing_clean
GROUP BY quality_label_binary;

SELECT anomaly_flag, COUNT(*) AS anomaly_count
FROM gold.secom_anomaly_predictions
GROUP BY anomaly_flag;

SELECT row_id, quality_label_binary, anomaly_score
FROM gold.secom_anomaly_predictions
ORDER BY anomaly_score ASC
LIMIT 20;

SELECT sensor, robust_shift
FROM gold.top_feature_shift_analysis
ORDER BY robust_shift DESC
LIMIT 10;
