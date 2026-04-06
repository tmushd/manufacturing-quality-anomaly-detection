-- Databricks SQL: schema setup
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Example Delta table creation templates
CREATE TABLE IF NOT EXISTS bronze.secom_features_raw
USING DELTA
AS SELECT * FROM VALUES (1) AS t(dummy)
WHERE 1 = 0;

CREATE TABLE IF NOT EXISTS bronze.secom_labels_raw
USING DELTA
AS SELECT * FROM VALUES (1) AS t(dummy)
WHERE 1 = 0;
