-- Sailing-time voyages completed on/before 31-Dec-2025
WITH voyages_base AS (
  SELECT
      v."VOYAGE_ID"                   AS voyage_id,
      v."VESSEL_ID"                   AS vessel_id,
      ve."VESSEL_TYPE_ID"             AS vessel_type_id,
      v."ESTIMATED_VOYAGE_START_DATE" AS voyage_start_ts,
      v."MILES_BALLAST"               AS miles_ballast,
      v."MILES_LOADED"                AS miles_loaded,
      NVL(v."MILES_BALLAST", 0) + NVL(v."MILES_LOADED", 0) AS miles_total,
      v."DAYS_TOTAL_AT_SEA"           AS days_at_sea,
      v."DAYS_TOTAL_IN_PORT"          AS days_in_port_total,
      v."DAYS_TOTAL"                  AS days_total,
      v."HAS_CANAL_PASSAGE"           AS has_canal_passage,
      v."CANAL_COST"                  AS canal_cost,
      v."CANAL_1_ID"                  AS canal_1_id,
      v."CANAL_2_ID"                  AS canal_2_id,
      v."VOYAGE_STATUS"               AS voyage_status
  FROM TBL_VOYAGE v
  LEFT JOIN TBL_VESSEL ve
    ON ve."VESSEL_ID" = v."VESSEL_ID"
  WHERE v."ESTIMATED_VOYAGE_START_DATE" IS NOT NULL
    AND v."DAYS_TOTAL_AT_SEA" IS NOT NULL
    AND v."DAYS_TOTAL_AT_SEA" > 0
    AND NVL(v."MILES_BALLAST", 0) + NVL(v."MILES_LOADED", 0) BETWEEN 20 AND 20000
    AND v."ESTIMATED_VOYAGE_START_DATE" <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
    AND v."ESTIMATED_VOYAGE_START_DATE" + NUMTODSINTERVAL(
          NVL(v."DAYS_TOTAL", v."DAYS_TOTAL_AT_SEA") * 24,
          'HOUR'
        ) <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
)
SELECT *
FROM voyages_base
ORDER BY voyage_start_ts;
