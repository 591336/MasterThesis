-- CANAL_EXPORT.sql
SELECT
  ca."CANAL_ID"   AS canal_id,
  ca."PORT_ID"    AS port_id,
  ca."EXTRA_TIME" AS extra_time_days
FROM TBL_CANAL ca
ORDER BY canal_id;
