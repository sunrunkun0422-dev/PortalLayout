# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 13; pairs: 78; components: [13]; predicted rooms: 13.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "21bd21f24927bb51151f1cc78b4f427e1d2a607e3df0e6eb3bb7754d120b5c52",
  "source_snapshot": "a73d0c205a4e5d630f3a4e47f43f928adaa7e992e1f9c23ad1a7937d1d384abe",
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
  },
  "source_files_verified": 28
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
  "frontend_seconds": 5.65536485100165,
  "encoding_seconds": 0.1558584530139342,
  "pair_seconds": 1.8235020699794404,
  "graph_seconds": 0.45182947401190177,
  "room_and_assembly_seconds": 0.4314703320269473,
  "total_seconds": 8.558203798020259,
  "peak_allocated_MiB": 638.47509765625,
  "observations_encoded": 13,
  "unordered_pairs": 78,
  "export_render_seconds": 0.3711907770484686
}
```
