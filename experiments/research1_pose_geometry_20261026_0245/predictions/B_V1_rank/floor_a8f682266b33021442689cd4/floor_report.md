# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 30; pairs: 435; components: [29, 1]; predicted rooms: 30.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "6fae13225a769741572ed3c16a1351c05be1f2d9bf6be5f3bf509b277723556f",
  "source_snapshot": "3f7e296e0a309780633430ba98ccaba7c455257eb410e722018d87670800d4ed",
  "inference_mode": "exploratory",
  "experimental_opening_threshold": 0.7,
  "room_association": "none",
  "pure_prediction_cache_reused": false,
  "pose_policy": "verifier_rank",
  "pose_workpoint_fingerprint": "63fa1a4ee3d97736109556c36b699f67753df5ccdc9224c593efe3a32354b4fb",
  "pose_geometry": {
    "pose_geometry": "local_m2",
    "scale_shrinkage": 0.0
  },
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
  "verifier_rejected_or_no_workpoint": 349
}
```

Runtime:
```json
{
  "frontend_seconds": 0.0,
  "encoding_seconds": 0.209204426035285,
  "pair_seconds": 5.255810390925035,
  "graph_seconds": 0.4471182479755953,
  "room_and_assembly_seconds": 3.2190945159527473,
  "total_seconds": 9.13216519501293,
  "peak_allocated_MiB": 91.89453125,
  "observations_encoded": 30,
  "unordered_pairs": 435,
  "export_render_seconds": 0.890533207973931
}
```
