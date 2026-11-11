# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 27; pairs: 351; components: [27]; predicted rooms: 26.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "21bd21f24927bb51151f1cc78b4f427e1d2a607e3df0e6eb3bb7754d120b5c52",
  "source_snapshot": "3f7e296e0a309780633430ba98ccaba7c455257eb410e722018d87670800d4ed",
  "inference_mode": "exploratory",
  "experimental_opening_threshold": 0.7,
  "room_association": "geometry",
  "pure_prediction_cache_reused": true,
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
  "no_usable_exploratory_hypothesis": 214
}
```

Runtime:
```json
{
  "frontend_seconds": 0.0,
  "encoding_seconds": 0.0,
  "pair_seconds": 0.0,
  "graph_seconds": 8.17677573201945,
  "room_and_assembly_seconds": 5.409645589010324,
  "total_seconds": 13.634423538984265,
  "peak_allocated_MiB": 12.21875,
  "observations_encoded": 0,
  "unordered_pairs": 351,
  "export_render_seconds": 1.056080740003381
}
```
