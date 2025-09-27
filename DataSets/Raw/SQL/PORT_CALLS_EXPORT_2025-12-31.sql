-- PORT_CALLS_EXPORT_2025-12-31.sql
SELECT
  pc."PORT_CALL_ID"       AS port_call_id,
  pc."VOYAGE_ID"          AS voyage_id,
  pc."PORT_ID"            AS port_id,
  pc."TERMINAL_ID"        AS terminal_id,
  pc."IS_BALLAST"         AS is_ballast,
  pc."COMMODITY_ID"       AS commodity_id,
  pc."DAYS_IN_PORT"       AS days_in_port,
  pc."DAYS_STOPPAGES"     AS days_stoppages,
  pc."DAYS_EXTRA_IN_PORT" AS days_extra_in_port
FROM TBL_PORT_CALL pc
WHERE pc."DAYS_IN_PORT" IS NOT NULL
  AND pc."DAYS_IN_PORT" BETWEEN 0.04 AND 10.0
  AND pc."VOYAGE_ID" IN (
        SELECT v."VOYAGE_ID"
        FROM TBL_VOYAGE v
        WHERE v."ESTIMATED_VOYAGE_START_DATE" IS NOT NULL
          AND v."ESTIMATED_VOYAGE_START_DATE" <= TO_TIMESTAMP('31.12.2025 23:59:59','DD.MM.YYYY HH24:MI:SS')
          AND v."DAYS_TOTAL" IS NOT NULL
          AND v."ESTIMATED_VOYAGE_START_DATE" + NUMTODSINTERVAL(v."DAYS_TOTAL"*24,'HOUR')
              <= TO_TIMESTAMP('31.12.2025 23:59:59','DD.MM.YYYY HH24:MI:SS')
      )
ORDER BY pc."VOYAGE_ID", pc."PORT_CALL_ID";
