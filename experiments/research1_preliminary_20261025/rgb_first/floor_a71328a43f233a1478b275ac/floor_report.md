# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 13; pairs: 78; components: [13]; predicted rooms: 13.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "21bd21f24927bb51151f1cc78b4f427e1d2a607e3df0e6eb3bb7754d120b5c52",
  "source_snapshot": "dc299c2e7e0f17edd8a5391e0253e3f626a87e8a75484f2f4dd9c58cd1ff6286",
  "inference_mode": "exploratory",
  "experimental_opening_threshold": 0.5,
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
  "no_usable_exploratory_hypothesis": 48
}
```

Runtime:
```json
{
  "frontend_seconds": 15.727734661020804,
  "encoding_seconds": 0.1564505440182984,
  "pair_seconds": 1.993083884997759,
  "graph_seconds": 0.14511569199385121,
  "room_and_assembly_seconds": 0.4389358530170284,
  "total_seconds": 18.502521728980355,
  "peak_allocated_MiB": 42.37939453125,
  "observations_encoded": 13,
  "unordered_pairs": 78,
  "export_render_seconds": 0.2951158190262504
}
```
