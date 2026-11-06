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
  "no_accepted_correspondence": 130,
  "pose_model_not_calibrated": 23,
  "insufficient_independent_pose_evidence": 23
}
```

Runtime:
```json
{
  "encoding_seconds": 0.30343304498819634,
  "pair_seconds": 2.7567562900076155,
  "graph_seconds": 0.002105807012412697,
  "room_and_assembly_seconds": 0.04932364699197933,
  "total_seconds": 3.111641385999974,
  "peak_allocated_MiB": null,
  "observations_encoded": 18,
  "unordered_pairs": 153,
  "export_render_seconds": 2.060321020981064
}
```
