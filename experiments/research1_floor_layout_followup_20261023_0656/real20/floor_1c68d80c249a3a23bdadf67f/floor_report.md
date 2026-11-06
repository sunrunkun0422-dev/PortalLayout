# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 20; pairs: 190; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 20.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "faae5c3a9c362772848b075dd0e195fbb39af0c4a2e7907abe387f21ad739f85",
  "source_snapshot": "71ce7873a103b527465d20f8976c01b2fcad2580b1f1074e6e12962ddfba3680",
  "frontend_contract": "41c206dcf5ca135304410431f3df0598885e1efd23f6d192157972c01439d639",
  "pair_training_status": "smoke_trained",
  "auxiliary_training_status": "initialized_untrained",
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
  "opening_model_not_calibrated": 190,
  "no_accepted_correspondence": 101,
  "pose_model_not_calibrated": 89,
  "insufficient_independent_pose_evidence": 89
}
```

Runtime:
```json
{
  "encoding_seconds": 0.2812361250107642,
  "pair_seconds": 2.6693144110031426,
  "graph_seconds": 0.0019723869918379933,
  "room_and_assembly_seconds": 0.051403852994553745,
  "total_seconds": 3.003947876015445,
  "peak_allocated_MiB": null,
  "observations_encoded": 20,
  "unordered_pairs": 190,
  "export_render_seconds": 2.1253338029782753
}
```
