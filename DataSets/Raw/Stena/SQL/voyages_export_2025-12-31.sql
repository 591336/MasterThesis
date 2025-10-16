-- Stena voyage export scoped to completed voyages before 31-Dec-2025.
WITH voyages_filtered AS (
  SELECT
      v."VOYAGE_ID"                   AS voyage_id,
      v."VESSEL_ID"                   AS vessel_id,
      v."ESTIMATED_VOYAGE_START_DATE" AS voyage_start_ts,
      v."MILES_BALLAST"               AS miles_ballast,
      v."MILES_LOADED"                AS miles_loaded,
      NVL(v."MILES_BALLAST", 0) + NVL(v."MILES_LOADED", 0) AS miles_total,
      v."SPEED_BALLAST"               AS speed_ballast,
      v."SPEED_LOADED"                AS speed_loaded,
      v."DAYS_TOTAL_AT_SEA"           AS days_total_at_sea,
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
    AND v."DAYS_TOTAL" IS NOT NULL
    AND v."ESTIMATED_VOYAGE_START_DATE" <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
    AND v."ESTIMATED_VOYAGE_START_DATE" + NUMTODSINTERVAL(v."DAYS_TOTAL" * 24, 'HOUR')
        <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
    AND NVL(v."DAYS_TOTAL_AT_SEA", 0) > 0
    AND (NVL(v."MILES_BALLAST", 0) + NVL(v."MILES_LOADED", 0)) BETWEEN 20 AND 12000
)
SELECT
    vf.*,
    ve."VESSEL_TYPE_ID" AS vessel_type_id
FROM voyages_filtered vf
LEFT JOIN TBL_VESSEL ve
  ON ve."VESSEL_ID" = vf.vessel_id
ORDER BY vf.voyage_start_ts;
