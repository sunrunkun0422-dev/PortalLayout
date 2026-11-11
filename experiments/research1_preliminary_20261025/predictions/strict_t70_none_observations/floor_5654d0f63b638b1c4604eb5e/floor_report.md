# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 20; pairs: 190; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 20.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "21bd21f24927bb51151f1cc78b4f427e1d2a607e3df0e6eb3bb7754d120b5c52",
  "source_snapshot": "3f7e296e0a309780633430ba98ccaba7c455257eb410e722018d87670800d4ed",
  "inference_mode": "strict",
  "experimental_opening_threshold": null,
  "room_association": "none",
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
{}
```

Runtime:
```json
{
  "frontend_seconds": 0.0,
  "encoding_seconds": 0.2384861590107903,
  "pair_seconds": 6.73905495100189,
  "graph_seconds": 0.0023679679725319147,
  "room_and_assembly_seconds": 0.0744374530040659,
  "total_seconds": 7.097179907024838,
  "peak_allocated_MiB": 52.38037109375,
  "observations_encoded": 20,
  "unordered_pairs": 190,
  "export_render_seconds": 3.376986377988942
}
```
