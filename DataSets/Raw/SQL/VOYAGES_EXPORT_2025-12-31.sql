-- VOYAGES_EXPORT_2024-12-31.sql
WITH v AS (
  SELECT
    v."VOYAGE_ID"                   AS voyage_id,
    v."VESSEL_ID"                   AS vessel_id,
    v."ESTIMATED_VOYAGE_START_DATE" AS voyage_start_ts,
    v."MILES_BALLAST"               AS miles_ballast,
    v."MILES_LOADED"                AS miles_loaded,
    NVL(v."MILES_BALLAST",0) + NVL(v."MILES_LOADED",0) AS miles_total,
    v."SPEED_BALLAST"               AS speed_ballast,
    v."SPEED_LOADED"                AS speed_loaded,
    v."DAYS_TOTAL_AT_SEA"           AS days_at_sea,
    v."DAYS_TOTAL_IN_PORT"          AS days_in_port_total,
    v."DAYS_TOTAL"                  AS days_total,
    v."DAYS_OFFHIRE"                AS days_offhire,
    v."HAS_CANAL_PASSAGE"           AS has_canal_passage,
    v."CANAL_COST"                  AS canal_cost,
    v."CANAL_1_ID"                  AS canal_1_id,
    v."CANAL_2_ID"                  AS canal_2_id,
    v."VOYAGE_STATUS"               AS voyage_status
  FROM TBL_VOYAGE v
  WHERE v."ESTIMATED_VOYAGE_START_DATE" IS NOT NULL
)
SELECT
  v.voyage_id,
  v.vessel_id,
  ve."VESSEL_TYPE_ID"               AS vessel_type_id,
  v.voyage_start_ts,
  v.miles_ballast,
  v.miles_loaded,
  v.miles_total,
  v.speed_ballast,
  v.speed_loaded,
  v.days_at_sea,
  v.days_in_port_total,
  v.days_total,
  v.days_offhire,
  v.has_canal_passage,
  v.canal_cost,
  v.canal_1_id,
  v.canal_2_id,
  v.voyage_status
FROM v
LEFT JOIN TBL_VESSEL ve
  ON ve."VESSEL_ID" = v.vessel_id
WHERE
  -- start no later than cutoff
  v.voyage_start_ts <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
  -- finishes by cutoff: start + days_total ≤ cutoff
  AND v.days_total IS NOT NULL
  AND v.voyage_start_ts + NUMTODSINTERVAL(v.days_total*24, 'HOUR')
      <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
  -- sane rows only
  AND v.days_at_sea > 0
  AND v.miles_total BETWEEN 20 AND 12000
ORDER BY v.voyage_start_ts;

select count(*) from TBL_VOYAGE;
