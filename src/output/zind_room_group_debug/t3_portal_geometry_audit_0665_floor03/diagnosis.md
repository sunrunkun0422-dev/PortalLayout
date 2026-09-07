# Candidate-local portal geometry audit

This is a CPU-only diagnostic. All M0-M4 hypotheses were frozen from saved predictions before train GT was attached.

## Global heading

Room04 all-edge fourth resultant is 0.110893, while the prior robust-subset confidence was 0.810574. Douglas-Peucker long-segment aggregation recovers a coherent global direction without GT.
Room05 shows the same confidence mismatch: all-edge resultant 0.150695 versus reported 0.799323.

## Uniform-method pair results

| pair | method | yaw error deg | scale | translation error | translation meter error |
|---|---|---:|---:|---:|---:|
| 0665_floor_03_pano_34_pano_37_3 | M0 | 69.72040034076463 | 2.1970054856108403 | 3.469674695662878 | 5.074391388880112 |
| 0665_floor_03_pano_34_pano_37_3 | M1 | 112.13378297265199 | 0.584047206199605 | 2.463381795240766 | 3.6026902997330223 |
| 0665_floor_03_pano_34_pano_37_3 | M2 | 4.356262604660249 | 0.7805968807956978 | 0.4108970923602736 | 0.600936067521022 |
| 0665_floor_03_pano_34_pano_37_3 | M3 | 23.649418313740465 | 0.5359620127617876 | 0.48926324057747766 | 0.7155463819087546 |
| 0665_floor_03_pano_34_pano_37_3 | M4 | 7.712438675163186 | 0.8434428763581241 | 0.45939685517035733 | 0.6718668608527002 |
| 0665_floor_03_pano_34_pano_30_4 | M0 | 75.05439766878945 | 3.8822650357388095 | 3.5499408399288654 | 5.191780443188595 |
| 0665_floor_03_pano_34_pano_30_4 | M1 | 65.29338733793342 | 4.094128906489212 | 3.7497193555658574 | 5.483956070113737 |
| 0665_floor_03_pano_34_pano_30_4 | M2 | 9.334188558173556 | 1.126504006697444 | 0.29163736484735203 | 0.42651898597495486 |
| 0665_floor_03_pano_34_pano_30_4 | M3 | 35.566358223675515 | 2.639688238904444 | 0.8543442950119153 | 1.249476597666763 |
| 0665_floor_03_pano_34_pano_30_4 | M4 | 9.434775440738367 | 1.0135383863878507 | 0.3636123443861654 | 0.5317822306368006 |

## Interpretation

- M0 is the saved frame-fixed mapper baseline.
- M1 changes only predicted global-heading estimation; it still maps intervals through a global Manhattan wall index.
- M2 fits one candidate-local line; its primary TLS/RANSAC variant is selected by predicted-only reliability.
- M3 checks left/right context and marks disagreement as corner ambiguity.
- M4 retains up to three predicted-only local wall segments; the primary is quality-ranked without GT.
- Width-ratio scale is unchanged in every pair pose.
- Negative pairs never generate a portal pose.

## Required findings

1. Room04 is a simple, finite 256-point polygon, so the heading failure is not self-intersection or duplicate topology. It has an all-edge fourth resultant of 0.110893, 121 turns above 60 degrees, and a high direction-histogram entropy. The dominant cause is noisy/multimodal adjacent-edge direction.
2. Room01 M1 post-hoc heading error is only 0.265269 degrees, but M1 01<-04 yaw error is 112.133783 degrees. Global heading therefore does not resolve interval-to-opposite-wall assignment after global Manhattan fitting.
3. Global sequential RANSAC recovers every room's modulo-90 heading to within 1.344668 degrees, yet its pair yaw errors remain 111.442444 and 66.244384 degrees. It does not repair wall-side.
4. M2 candidate-local fitting reduces pair yaw errors to 4.356263 and 9.334189 degrees without GT selection.
5. Candidate-local tangent estimates are consistently close to GT, while endpoint midpoint/width errors remain candidate dependent; this separates orientation recovery from portal depth/extent quality.
6. M2 improves both positive edges simultaneously. M4's predicted-only primary also improves both, but one primary segment has only four support points.
7. M2 width-ratio scales are 0.780597 and 1.126504; M4 gives 0.843443 and 1.013538.
8. M2 translation errors fall to 0.410897 and 0.291637 coordinate units.
9. Useful predicted-only failure signals are: all-edge fourth resultant, global mean axis residual, local normalized residual, angular dispersion, context angle/offset disagreement, support count, ray condition, and width/radius. The old confidence is misleading because it is normalized only over the robustly reweighted subset.
10. The evidence favors candidate-local portal geometry with retained multi-wall hypotheses for ambiguous contexts. Global heading alone is insufficient; portal depth/extent and quality ranking should be audited next before any formal replacement.

See audit_summary.json and portal_hypotheses.csv for the post-hoc GT details; no GT metric participates in hypothesis selection.
