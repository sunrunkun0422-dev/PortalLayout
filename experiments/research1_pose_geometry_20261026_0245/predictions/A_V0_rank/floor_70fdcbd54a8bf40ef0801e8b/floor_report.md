# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 18; pairs: 153; components: [18]; predicted rooms: 18.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "16dd6a6dc79dbf4c60c9065576c8a7f0709eddb2e14bcf6c1e60d1f0397b7f55",
  "source_snapshot": "3f7e296e0a309780633430ba98ccaba7c455257eb410e722018d87670800d4ed",
  "inference_mode": "exploratory",
  "experimental_opening_threshold": 0.7,
  "room_association": "none",
  "pure_prediction_cache_reused": false,
  "pose_policy": "verifier_rank",
  "pose_workpoint_fingerprint": "a02748b87115f5fd574e3cb6b5b0f65355e011356b27eae43105e3e9ee117711",
  "pose_geometry": {
    "pose_geometry": "dominant_wall",
    "scale_shrinkage": 1.0
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
  "verifier_rejected_or_no_workpoint": 101
}
```

Runtime:
```json
{
  "frontend_seconds": 0.0,
  "encoding_seconds": 0.13490465003997087,
  "pair_seconds": 1.9224762679659761,
  "graph_seconds": 0.16738418204477057,
  "room_and_assembly_seconds": 0.5709529389860108,
  "total_seconds": 2.79612218902912,
  "peak_allocated_MiB": 73.8330078125,
  "observations_encoded": 18,
  "unordered_pairs": 153,
  "export_render_seconds": 0.7272095590014942
}
```
