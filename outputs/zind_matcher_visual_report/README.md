# T3 Matcher five validation cases

All figures use existing frozen validation predictions and RGB. No training, model loading, GPU forward or test access.

Use t3_matcher_five_cases.png/pdf/svg for the 16:9 overview, or t3_matcher_case_slides.pdf for five full-size pages. Individual cases also have PNG/PDF/SVG versions.

Full panoramas preserve their 2:1 ratio; detail crops are centered on the predicted final choice (raw Rank-1 when rejected), with cyclic display across the seam. GT never determines the crop or prediction.

No-match scores and real-real scores are not complementary calibrated probabilities. Acceptance is the original bidirectional dustbin decision, not a threshold on the displayed no-match scalar.

| Case | Pair ID | Raw #1 | Final | GT rank |
| --- | --- | --- | --- | --- |
| correct_match | 0461_floor_02_pano_76_pano_83_0 | A1 ↔ B0 | A1 ↔ B0 | 1 |
| dustbin_reject | 0417_floor_01_pano_45_pano_43_14 | A0 ↔ B0 | REJECT / No Match | 1 |
| wrong_selection | 0220_floor_01_pano_40_pano_37_13 | A2 ↔ B0 | A2 ↔ B0 | 5 |
| negative_correct_reject | 0003_floor_01_pano_95_pano_34_negative | A1 ↔ B0 | REJECT / No Match | None |
| negative_false_accept | 0293_floor_01_pano_36_pano_30_negative | A1 ↔ B0 | A1 ↔ B0 | None |

case_selection.json contains all three alternatives per category, selection reasons, prior positive visual-review uncertainty, and source SHA-256 values. candidate_review/ contains their rendered panels.

These are selected qualitative examples, not aggregate performance or causal proof. Wrong-selection case 0220 retains the prior audit flag of possible annotation ambiguity; its frozen classification is unambiguous under the existing evaluation protocol.

Run: `env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/feixia/anaconda3/envs/bi_layout/bin/python -B tools/visualize_zind_matcher_five_cases.py`
