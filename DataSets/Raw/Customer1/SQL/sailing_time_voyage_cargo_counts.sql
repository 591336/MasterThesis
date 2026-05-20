SELECT
    c."VOYAGE_ID"                   AS voyage_id,
    COUNT(*)                        AS n_cargo_rows,
    COUNT(DISTINCT c."CARGO_ID")    AS n_unique_cargo,
    SUM(NVL(c."CARGO_QUANTITY",0))  AS total_cargo_quantity
FROM TBL_CARGO c
WHERE c."VOYAGE_ID" IN (
        SELECT v."VOYAGE_ID"
        FROM TBL_VOYAGE v
        WHERE v."ESTIMATED_VOYAGE_START_DATE" IS NOT NULL
          AND v."DAYS_TOTAL_AT_SEA" IS NOT NULL
          AND v."DAYS_TOTAL_AT_SEA" > 0
          AND v."ESTIMATED_VOYAGE_START_DATE" <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
          AND v."ESTIMATED_VOYAGE_START_DATE" + NUMTODSINTERVAL(
                NVL(v."DAYS_TOTAL", v."DAYS_TOTAL_AT_SEA") * 24,
                'HOUR'
              ) <= TO_TIMESTAMP('31.12.2025 23:59:59', 'DD.MM.YYYY HH24:MI:SS')
      )
GROUP BY c."VOYAGE_ID"
ORDER BY voyage_id;
