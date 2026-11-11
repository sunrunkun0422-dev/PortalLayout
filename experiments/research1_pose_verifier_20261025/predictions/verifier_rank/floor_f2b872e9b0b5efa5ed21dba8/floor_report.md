# Floor reconstruction

Status: exploratory_multi_observation_reconstruction

Coordinates: {"frame": "bi_component_xz", "units": "camera_height_normalized", "scale_status": "unknown", "transform_direction": "local_to_component"}

Inputs: 38; pairs: 703; components: [37, 1]; predicted rooms: 38.

Independent components have unknown relative positions. Sidebar placement is display-only.

Calibration and training:
```json
{
  "weights_hash": "34cb98ea45e8530da76cca476e2efa409168d1badd24e3596b9c257e0ca83dd8",
  "source_snapshot": "3f7e296e0a309780633430ba98ccaba7c455257eb410e722018d87670800d4ed",
  "inference_mode": "exploratory",
  "experimental_opening_threshold": 0.7,
  "room_association": "none",
  "pure_prediction_cache_reused": false,
  "pose_policy": "verifier_rank",
  "pose_workpoint_fingerprint": "a591ffb758323d9fcae5cc322406dd6fbc21c32cc2467ccf83d0aa28dd0844c7",
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
  "verifier_rejected_or_no_workpoint": 577
}
```

Runtime:
```json
{
  "frontend_seconds": 0.0,
  "encoding_seconds": 0.47477039397927,
  "pair_seconds": 4.260444045008626,
  "graph_seconds": 0.9245466879801825,
  "room_and_assembly_seconds": 7.230288117018063,
  "total_seconds": 12.891634258034173,
  "peak_allocated_MiB": 79.53515625,
  "observations_encoded": 38,
  "unordered_pairs": 703,
  "export_render_seconds": 1.0239383080042899
}
```
