# ZInD-BiPair-v1

ZInD-BiPair-v1 is a pair-centered derivative of ZInD for Bi-Layout
cross-panorama opening matching and layout merging. Version 1 contains strict
`partial_opening` pairs only: both views come from different partial rooms in
the same complete room, and positive pairs share a matched `layout_raw.openings`
segment.

The dataset does not copy panorama images. `view_A.image_path` and
`view_B.image_path` are relative to the native ZInD data root recorded in
`dataset_info.json`.

## Files

- `manifests/{train,val,test}_pairs.jsonl`: pair metadata and geometry.
- `labels/<split>/<pair_id>.npz`: 256-token Bi-Layout, opening, affinity,
  relative-pose, and complete-layout targets.
- `statistics/*_stats.json`: split counts and portal matching diagnostics.
- `statistics/invalid_pairs.jsonl`: filtered views/pairs and reasons.

## Scope

- Primary panoramas only.
- `is_inside=true` and `is_ceiling_flat=true`.
- `layout_raw`, `layout_visible`, `layout_complete`, metric floor scale, and
  `floor_plan_transformation` are required.
- House-level train/val/test partitions are inherited from ZInD.
