-- DISTANCES_EXPORT.sql
SELECT
  d."DISTANCE_ID" AS distance_id,
  d."F_PORT_ID"   AS from_port_id,
  d."T_PORT_ID"   AS to_port_id,
  d."MILES"       AS miles
FROM TBL_DISTANCE d
WHERE d."MILES" IS NOT NULL
ORDER BY from_port_id, to_port_id;
