# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 20; pairs: 190; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 20.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "4a097e3e56a45092c40d0b80d5a87565ba8b91640d710b6b3d18aeb421da511c",
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
  "encoding_seconds": 0.8548784340091515,
  "pair_seconds": 3.0702364550088532,
  "graph_seconds": 0.0020711399847641587,
  "room_and_assembly_seconds": 0.056436145008774474,
  "total_seconds": 3.9836714139964897,
  "peak_allocated_MiB": 42.85009765625,
  "observations_encoded": 20,
  "unordered_pairs": 190,
  "export_render_seconds": 2.2304092799895443
}
```
