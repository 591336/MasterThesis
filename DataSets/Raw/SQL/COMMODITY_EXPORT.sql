-- COMMODITY_EXPORT.sql
SELECT
  c."COMMODITY_ID"     AS commodity_id,
  c."COMMODITY_CODE"   AS commodity_code,
  c."COMMODITY_NAME"   AS commodity_name,
  c."COMMODITY_GROUP_ID" AS commodity_group_id
FROM TBL_COMMODITY c
ORDER BY commodity_id;
