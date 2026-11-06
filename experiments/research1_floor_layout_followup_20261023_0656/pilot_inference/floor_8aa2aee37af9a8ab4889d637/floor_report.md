# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 17; pairs: 136; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 17.

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
  "opening_model_not_calibrated": 136,
  "no_accepted_correspondence": 87,
  "pose_model_not_calibrated": 49,
  "insufficient_independent_pose_evidence": 49
}
```

Runtime:
```json
{
  "encoding_seconds": 0.2353508009982761,
  "pair_seconds": 2.3399537650111597,
  "graph_seconds": 0.0019482510106172413,
  "room_and_assembly_seconds": 0.0718944999971427,
  "total_seconds": 2.6491683089989237,
  "peak_allocated_MiB": null,
  "observations_encoded": 17,
  "unordered_pairs": 136,
  "export_render_seconds": 2.0606588640075643
}
```
