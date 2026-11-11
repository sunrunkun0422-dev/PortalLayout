# Floor reconstruction

Status: isolated_observations_only

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 22; pairs: 231; components: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; predicted rooms: 22.

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
  "encoding_seconds": 0.25919457798590884,
  "pair_seconds": 8.311146048014052,
  "graph_seconds": 0.0025513669825159013,
  "room_and_assembly_seconds": 0.06848029000684619,
  "total_seconds": 8.703134924988262,
  "peak_allocated_MiB": 55.40087890625,
  "observations_encoded": 22,
  "unordered_pairs": 231,
  "export_render_seconds": 3.5466486089862883
}
```
