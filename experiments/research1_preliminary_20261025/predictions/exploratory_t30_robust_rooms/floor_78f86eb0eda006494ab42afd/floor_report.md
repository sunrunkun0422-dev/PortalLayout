# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 24; pairs: 276; components: [23, 1]; predicted rooms: 23.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "21bd21f24927bb51151f1cc78b4f427e1d2a607e3df0e6eb3bb7754d120b5c52",
  "source_snapshot": "3f7e296e0a309780633430ba98ccaba7c455257eb410e722018d87670800d4ed",
  "inference_mode": "exploratory",
  "experimental_opening_threshold": 0.3,
  "room_association": "geometry",
  "pure_prediction_cache_reused": false,
  "matcher_type": "new",
  "frontend_contract": "5f8009ac1e7925813ee86246f169851af359636389e99f35721c8a7855ed9e89",
  "pair_training_status": "trained",
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
  "no_usable_exploratory_hypothesis": 221
}
```

Runtime:
```json
{
  "frontend_seconds": 0.0,
  "encoding_seconds": 0.2883925700443797,
  "pair_seconds": 9.377048030029982,
  "graph_seconds": 1.4113787619862705,
  "room_and_assembly_seconds": 2.6868131029768847,
  "total_seconds": 13.818863090011291,
  "peak_allocated_MiB": 58.36865234375,
  "observations_encoded": 24,
  "unordered_pairs": 276,
  "export_render_seconds": 0.6854831880191341
}
```
