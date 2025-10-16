-- Stena port-call export with voyage completion guardrails and optional terminal name.
WITH in_scope_voyages AS (
  SELECT
      v."VOYAGE_ID"
  FROM TBL_VOYAGE v
  WHERE v."ESTIMATED_VOYAGE_START_DATE" IS NOT NULL
    AND v."DAYS_TOTAL" IS NOT NULL
    AND v."ESTIMATED_VOYAGE_START_DATE" <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
    AND v."ESTIMATED_VOYAGE_START_DATE" + NUMTODSINTERVAL(v."DAYS_TOTAL" * 24, 'HOUR')
        <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
)
SELECT
    pc."PORT_CALL_ID"       AS port_call_id,
    pc."VOYAGE_ID"          AS voyage_id,
    pc."PORT_ID"            AS port_id,
    pc."TERMINAL_ID"        AS terminal_id,
    vw."TERMINAL_NAME"      AS terminal_name,
    pc."IS_BALLAST"         AS is_ballast,
    pc."COMMODITY_ID"       AS commodity_id,
    pc."DAYS_IN_PORT"       AS days_in_port,
    pc."DAYS_STOPPAGES"     AS days_stoppages,
    pc."DAYS_EXTRA_IN_PORT" AS days_extra_in_port
FROM TBL_PORT_CALL pc
JOIN in_scope_voyages v ON v."VOYAGE_ID" = pc."VOYAGE_ID"
LEFT JOIN VW_BASIC_PORT_CALL_DATA vw ON vw."PORT_CALL_ID" = pc."PORT_CALL_ID"
WHERE pc."DAYS_IN_PORT" BETWEEN 0.04 AND 10.0
ORDER BY pc."VOYAGE_ID", pc."PORT_CALL_ID";
