"""Versioned JSON schemas; no model, checkpoint, or dataset access at import."""

ID = {"type": "string", "minLength": 1}
RGB = {"type": "object", "required": ["root_key", "relative_path"],
       "properties": {"root_key": {"const": "zind_raw"}, "relative_path": ID},
       "additionalProperties": False}
OBSERVATION = {"type": "object", "required": ["observation_id", "rgb", "image_size_wh"],
               "properties": {"observation_id": ID, "rgb": RGB,
                              "image_size_wh": {"type": "array", "items": {"type": "integer", "minimum": 1},
                                                "minItems": 2, "maxItems": 2}},
               "additionalProperties": False}
INPUT_PROPERTIES = {
    "schema_version": {"const": "zind-floorset.input.v1"}, "sample_id": ID, "floor_uid": ID,
    "task": {"const": "unposed_floor_layout_from_panoramas"},
    "input_mode": {"const": "panorama_only"},
    "selection_policy": {"enum": ["all_panoramas", "primary_only", "one_per_space", "sampled_N_views"]},
    "projection": {"const": "equirectangular"}, "pose_prior": {"const": "none"},
    "calibration_mode": {"const": "scale_unknown"},
    "observation_order": {"type": "array", "items": ID, "uniqueItems": True},
    "observations": {"type": "array", "items": OBSERVATION},
}
INPUT_SCHEMA = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                "required": list(INPUT_PROPERTIES), "properties": INPUT_PROPERTIES,
                "additionalProperties": False}


def record_schema(version, fields):
    return {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
            "required": ["schema_version"] + fields,
            "properties": {"schema_version": {"const": version}}}


SCHEMAS = {
    "input": INPUT_SCHEMA,
    "floor_gt": record_schema("zind-floorset.floor-gt.v1", [
        "floor_uid", "house_id", "floor_id", "source_family", "coordinate_spec",
        "metric_scale", "metric_valid", "complete_rooms", "partial_spaces",
        "observations_membership", "structural_walls", "junctions", "interfaces",
        "room_adjacency", "evaluation_masks", "issues"]),
    "observation_gt": record_schema("zind-floorset.observation-gt.v1", [
        "observation_id", "source_ref", "complete_room_uid", "partial_space_uid",
        "layouts", "interfaces_by_layout", "pose_gt_available", "G_local_to_floor_gt",
        "coordinate_spec", "supervision_masks"]),
    "pair_gt": record_schema("zind-floorset.pair-gt.v1", [
        "pair_id", "observation_a", "observation_b", "relation_scope", "same_room_gt",
        "physical_portal_label", "visual_portal_label", "portal_correspondences",
        "covisibility", "pose_gt_available", "pose_supervision_valid", "T_B_to_A_gt",
        "supervision_masks", "coordinate_spec"]),
}

# Offline records allow provenance extensions, but constrain core field types.
for name, array_fields in {
    "floor_gt": ["complete_rooms", "partial_spaces", "observations_membership", "structural_walls",
                 "junctions", "interfaces", "room_adjacency", "issues"],
    "observation_gt": [], "pair_gt": ["portal_correspondences"],
}.items():
    for field in array_fields:
        SCHEMAS[name]["properties"][field] = {"type": "array"}
for name in ("floor_gt", "observation_gt", "pair_gt"):
    for field in SCHEMAS[name]["required"]:
        if field.endswith("_uid") or field in ("observation_id", "pair_id", "observation_a", "observation_b"):
            SCHEMAS[name]["properties"][field] = ID
SCHEMAS["floor_gt"]["properties"].update({
    "metric_scale": {"type": ["number", "null"], "exclusiveMinimum": 0},
    "metric_valid": {"type": "boolean"}, "source_family": {"const": "merger"},
    "evaluation_masks": {"type": "object"}})
for name in ("observation_gt", "pair_gt"):
    SCHEMAS[name]["properties"]["pose_gt_available"] = {"type": "boolean"}
    SCHEMAS[name]["properties"]["supervision_masks"] = {"type": "object"}
SCHEMAS["pair_gt"]["properties"].update({
    "physical_portal_label": {"enum": ["shared", "no_shared", "unknown"]},
    "visual_portal_label": {"enum": ["positive", "negative", "unknown"]},
    "pose_supervision_valid": {"type": "boolean"}})
