# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 30; pairs: 435; components: [30]; predicted rooms: 30.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "34cb98ea45e8530da76cca476e2efa409168d1badd24e3596b9c257e0ca83dd8",
  "source_snapshot": "3f7e296e0a309780633430ba98ccaba7c455257eb410e722018d87670800d4ed",
  "inference_mode": "exploratory",
  "experimental_opening_threshold": 0.7,
  "room_association": "none",
  "pure_prediction_cache_reused": true,
  "pose_policy": "geometry",
  "pose_workpoint_fingerprint": null,
  "matcher_type": "new",
  "frontend_contract": "5f8009ac1e7925813ee86246f169851af359636389e99f35721c8a7855ed9e89",
  "pair_training_status": "trained",
  "auxiliary_training_status": "trained",
  "calibration_track": null,
  "calibration_status": {
    "opening": false,
    "pose": false,
    "room": false
  }
}
```

Reasons:
```json
{
  "no_usable_exploratory_hypothesis": 347
}
```

Runtime:
```json
{
  "frontend_seconds": 0.0,
  "encoding_seconds": 0.0,
  "pair_seconds": 0.0,
  "graph_seconds": 0.8224916079780087,
  "room_and_assembly_seconds": 1.5316823199973442,
  "total_seconds": 2.404398111975752,
  "peak_allocated_MiB": 12.21875,
  "observations_encoded": 0,
  "unordered_pairs": 435,
  "export_render_seconds": 0.5379477189853787
}
```
