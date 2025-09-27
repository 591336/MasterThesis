-- OFFHIRE_EXPORT_ASOF_2024-12-31.sql
SELECT
  o."OFFHIRE_ID"         AS offhire_id,
  o."VOYAGE_HEADER_ID"   AS voyage_header_id,
  o."VESSEL_CODE_ID"     AS vessel_code_id,
  o."PORT_ID"            AS port_id,
  o."OFFHIRE_START_DATE" AS offhire_start_ts,
  o."OFFHIRE_END_DATE"   AS offhire_end_ts,
  o."OFFHIRE_DAYS"       AS offhire_days,
  o."OFFHIRE_COST"       AS offhire_cost
FROM TBL_OFFHIRE o
WHERE o."OFFHIRE_START_DATE" <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
ORDER BY offhire_start_ts NULLS LAST;


select * from tbl_offhire;