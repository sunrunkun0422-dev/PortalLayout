# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 20; pairs: 190; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 20.

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
  "opening_model_not_calibrated": 190,
  "pose_model_not_calibrated": 66,
  "insufficient_independent_pose_evidence": 66,
  "no_accepted_correspondence": 124
}
```

Runtime:
```json
{
  "encoding_seconds": 0.32077902599121444,
  "pair_seconds": 2.969725278991973,
  "graph_seconds": 0.032841431006090716,
  "room_and_assembly_seconds": 0.05812933400738984,
  "total_seconds": 3.381498285016278,
  "peak_allocated_MiB": null,
  "observations_encoded": 20,
  "unordered_pairs": 190,
  "export_render_seconds": 2.29829950700514
}
```
