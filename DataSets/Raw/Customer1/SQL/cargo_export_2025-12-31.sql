-- Cargo legs linked to voyages inside the modelling window (trimmed column set).
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
    c."VOYAGE_ID"           AS voyage_id,
    c."CARGO_ID"            AS cargo_id,
    c."COMMODITY_ID"        AS commodity_id,
    c."BASELINE_TERM_ID"    AS baseline_term_id,
    c."BASELINE_TERM_2_ID"  AS baseline_term_2_id,
    c."LAYCAN_FROM"         AS laycan_from,
    c."LAYCAN_TO"           AS laycan_to,
    c."CARGO_QUANTITY"      AS cargo_quantity,
    c."QUANTITY_OPTION_PCT" AS quantity_option_pct,
    c."BOOKED_QUANTITY"     AS booked_quantity
FROM TBL_CARGO c
JOIN in_scope_voyages v ON v."VOYAGE_ID" = c."VOYAGE_ID"
ORDER BY c."VOYAGE_ID", c."CARGO_ID";
