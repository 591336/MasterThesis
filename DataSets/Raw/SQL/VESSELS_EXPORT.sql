-- VESSELS_EXPORT.sql
SELECT
  ve."VESSEL_ID"          AS vessel_id,
  ve."VESSEL_NAME"        AS vessel_name,
  ve."VESSEL_TYPE_ID"     AS vessel_type_id,
  ve."IMO_NUMBER"         AS imo_number,

  -- capacities / dimensions
  ve."DWT_SUMMER"         AS dwt_summer,
  ve."DRAFT_SUMMER"       AS draft_summer,
  ve."LOA"                AS loa,
  ve."BEAM"               AS beam,
  ve."AIR_DRAFT"          AS air_draft,

  -- build / registry
  ve."BUILT_YEAR"         AS built_year,
  ve."FLAG_ID"            AS flag_id,
  ve."REGISTERED_PORT_ID" AS registered_port_id,

  -- useful operational flags (optional)
  ve."ICE_CLASS_ID"       AS ice_class_id,
  ve."SCRUBBER_FITTED_ME" AS scrubber_fitted_me
FROM TBL_VESSEL ve
WHERE ve."VESSEL_NAME" IS NOT NULL
ORDER BY vessel_id;
