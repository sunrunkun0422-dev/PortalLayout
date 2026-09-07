# ZInD val geometry smoke preflight

validation smoke test / engineering preflight; not a scientific final result

## Completion and safety

- completed: 64/64
- split: val
- test data accessed: false
- one frozen model/matcher prediction fans out to G0/G1/G2

## Matching

- TP/TN/FP/FN: 7/30/2/25
- positive recall: 0.2188
- negative rejection: 0.9375
- end-to-end candidate top1: 0.0625
- matcher maximum branch difference: 0.0

## Evaluation strata

- S0_all_pairs: 64
- S1_all_gt_positive: 32
- S2_gt_positive_predicted_match: 7
- S3_gt_positive_selected_candidate_correct: 2
- S4_gt_positive_m2_geometry_valid: 7

## Geometry on S3 (correct selected candidate)

| mode | pose coverage | yaw median | translation median (m) | log-scale median |
|---|---:|---:|---:|---:|
| legacy_polygon | 2/2 | 13.15168697925001 | 3.5690888500123243 | 0.8523107144200018 |
| frame_polygon | 2/2 | 50.745302240792185 | 3.3580644321693027 | 0.5437582840099647 |
| candidate_local_m2 | 2/2 | 3.8514446527294743 | 3.2111224895659776 | 1.716204359643093 |

## M2 quality

- valid over all positives: 7/32
- hard failures: 0
- soft-fail pairs: 4
- variants: {'ransac': 5, 'weighted_tls': 9}

## Paired S3 comparison (M2 - G1)

- yaw W/T/L: 2/0/0
- translation W/T/L: 1/0/1
- log-scale W/T/L: 0/0/2

No confidence interval or significance claim is made for this 64-pair smoke set.
