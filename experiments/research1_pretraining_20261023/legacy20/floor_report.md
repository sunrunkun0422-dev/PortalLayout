# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 20; pairs: 190; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 20.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "30809362947737c0b04a64bb0b35b496171164743cd18dce2f1efc1fc9643fc5",
  "source_snapshot": "71ce7873a103b527465d20f8976c01b2fcad2580b1f1074e6e12962ddfba3680",
  "matcher_type": "legacy",
  "frontend_contract": "41c206dcf5ca135304410431f3df0598885e1efd23f6d192157972c01439d639",
  "pair_training_status": "initialized_untrained",
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
  "pose_model_not_calibrated": 190,
  "insufficient_independent_pose_evidence": 190
}
```

Runtime:
```json
{
  "encoding_seconds": 0.2819898880261462,
  "pair_seconds": 5.344292838999536,
  "graph_seconds": 0.002002188004553318,
  "room_and_assembly_seconds": 0.05437022401019931,
  "total_seconds": 5.682920838007703,
  "peak_allocated_MiB": null,
  "observations_encoded": 20,
  "unordered_pairs": 190,
  "export_render_seconds": 2.20449054898927
}
```
