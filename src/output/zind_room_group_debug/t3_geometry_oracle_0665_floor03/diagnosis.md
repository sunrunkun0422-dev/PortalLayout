# T3 three-room geometry oracle diagnosis

## Executive conclusion

The frozen matcher/opening selection is not the failure source. Both edges recover the exact GT yaw when the same predicted interval is projected onto the GT panorama-local polygon (C2). The first irreversible error is the fixed-axis Manhattan polygon being treated as if it still retained each panorama's local heading; interval-to-wall projection then attaches physical doors to incorrect canonical walls. Door-width ratio has a valid Sim(2) interpretation for ideal geometry, but the raw predicted widths are unstable and amplify translation errors; rotation/frame loss must be repaired first.

## Key numeric chain

| Pair | C0 yaw err | C1 yaw err | C2 yaw err | C3 yaw err | C4 yaw err | C0 scale | C4 width scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0665_floor_03_pano_34_pano_37_3 | 101.178516 | 101.178516 | 0.000000 | 0.000000 | 0.000000 | 0.548843 | 1.024043 |
| 0665_floor_03_pano_34_pano_30_4 | 78.364440 | 11.635560 | 0.000000 | 0.000000 | 0.000000 | 3.391321 | 1.033316 |

## Required questions

1. **Can GT polygon + GT endpoints recover GT pose with the current formula?** It recovers GT yaw to numerical precision and translation within a small annotation-level residual. Width-ratio scale is not exact because the two independently annotated door widths differ; C4 scales are 1.024043 and 1.033316 although GT scale is 1.
2. **Where does yaw first become a multiple of 90 degrees?** In `prediction_to_layout -> post_process(type_name='manhattan') -> fit_layout`, before candidate geometry is built. The mapper and joint layout preserve/consume those axis-aligned wall tangents; they do not create the lost arbitrary heading.
3. **Was per-panorama canonical-to-local heading lost?** Yes in the retained geometry contract. T3 JSON contains only the post-Manhattan layout and no pre-Manhattan polygon/raw depth heading. The train GT rotations are arbitrary and differ per panorama.
4. **Where is the loss?** Primarily Manhattan postprocessing/export. It becomes observable in candidate mapping because the mapper assumes that polygon rays are panorama-local. Joint layout then deterministically fits the already-wrong endpoints.
5. **Does GT interval through the current mapper recover the opening?** On the GT polygon (C3), yes: both yaw errors are numerical zero and translation residuals are small. On the predicted polygon (C1), no; one small wrap-interval change even switches the dominant wall.
6. **Does predicted interval remain wrong on the GT polygon?** No. C2 recovers exact GT yaw on both edges, so the frozen predicted intervals are sufficient for these portals.
7. **Does predicted polygon remain wrong with GT interval?** Yes. C1 remains wrong, demonstrating that interval supervision alone cannot repair the canonicalized polygon/wall association.
8. **Is unit scale better than width ratio?** For this fixed group it always eliminates scale-label error because GT scale is 1. For C0 it strongly improves translation on the 01-05 edge but cannot repair rotation/frame errors; for C2-C4 width fitting can slightly reduce translation by absorbing endpoint-annotation mismatch. Across the full manifest, width ratio is theoretically consistent with the rare non-unit Sim(2) labels, so unit scale is an oracle for this group rather than a universal replacement.
9. **Is train/val relative scale essentially one?** Statistically yes but contractually not always. 16340/16568 pose-valid positives are within 1e-9 of 1 (ratio 0.986239); 228 are non-unit, with range 0.443511--2.535827 and |log(scale)| max 9.305e-01. The manifest is therefore Sim(2), overwhelmingly concentrated on an SE(2) mode.
10. **What dominates translation error?** C0 translation inherits both wrong rotation/wall placement and extreme width scale. C2 reduces translation to a small residual without changing the predicted interval, proving polygon frame/wall geometry is primary; unit scale is the second correction. Opening midpoint fitting itself is exact by construction and is not the primary source.
11. **Where should the minimal formal repair live?** Preserve or restore panorama heading around `tools/debug_cross_scene_flow.py::prediction_to_layout` (or its `post_process/fit_layout` helper) so `opening_candidates_from_intervals` receives a true panorama-local polygon. For this fixed unit-scale group, `calibrate_scale=False` is a valid diagnostic baseline. A global change in `utils/joint_layout.py::_fit_similarity` must instead retain the rare non-unit Sim(2) labels through a robust/clamped estimator or an explicitly supervised scale source. Do not patch candidate ranks or add a verifier.
12. **Does this require retraining?** The demonstrated frame/wall failure is deterministic parsing geometry and does not require retraining. A heading-preserving postprocess and fixed-group unit-scale validation should come first. Only a learned replacement for the full-dataset scale estimator would require additional training.

## Important wall-association evidence

For room01, the two labeled shared doors lie on the same GT complete-layout wall family. In C0 they map to predicted wall 2 and predicted wall 8, which are opposite parallel walls. A single global heading cannot repair that wall-side swap; the panorama-local polygon/ray contract must be corrected before fusion.

All oracle rows have `oracle_only=true` and `valid_for_inference=false`. Only the exact C0/current/width-ratio row selected by the existing centroid-side rule is marked as the frozen formal path.
