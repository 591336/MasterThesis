-- Commodity reference table used for enrichment.
SELECT
    cm."COMMODITY_ID"       AS commodity_id,
    cm."COMMODITY_GROUP_ID" AS commodity_group_id,
    cm."COMMODITY_CODE"     AS commodity_code,
    cm."COMMODITY_NAME"     AS commodity_name
FROM TBL_COMMODITY cm
ORDER BY commodity_id;
