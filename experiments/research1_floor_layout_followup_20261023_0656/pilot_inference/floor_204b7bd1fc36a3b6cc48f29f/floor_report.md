# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 18; pairs: 153; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 18.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "322c6fb1590c27663dea997347aa2a428de04dbe2b6bc28dbe5a3f9514b2e819",
  "source_snapshot": "f2b1a89c580c662cd289c3b5e71dc69503bf019750ec9e7a2619e8f7b510d2ac",
  "frontend_contract": "1ad32b6476d34f4e06cc1257f60e00562d181ee16ddeee24d3a88a414b6a563d",
  "pair_training_status": "trained",
  "auxiliary_training_status": "trained",
  "calibration_track": "geometry_dev",
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
  "opening_model_not_calibrated": 153,
  "no_accepted_correspondence": 103,
  "pose_model_not_calibrated": 50,
  "insufficient_independent_pose_evidence": 50
}
```

Runtime:
```json
{
  "encoding_seconds": 0.2615857170021627,
  "pair_seconds": 2.0349533160042483,
  "graph_seconds": 0.0020977690001018345,
  "room_and_assembly_seconds": 0.05866515898378566,
  "total_seconds": 2.3573247629974503,
  "peak_allocated_MiB": null,
  "observations_encoded": 18,
  "unordered_pairs": 153,
  "export_render_seconds": 2.013567051006248
}
```
