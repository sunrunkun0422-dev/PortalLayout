import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "visual_audit_zind_candidate_failures",
    ROOT / "tools/visual_audit_zind_candidate_failures.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def audit_row(reason, *, present=False, predicted=False, correct=False, index=0):
    return {
        "pair_id": "pair_{:04d}".format(index),
        "house_id": "house_{:04d}".format(index),
        "split": "val",
        "test_data_accessed": False,
        "primary_pair_reason": reason,
        "legacy": {
            "target_present": present,
            "predicted_match": predicted,
            "selected_correct": correct,
        },
    }


def exact_full_val_rows():
    rows = []
    index = 0
    for count, kwargs in (
        (501, {"reason": "target_present_correct", "present": True, "predicted": True, "correct": True}),
        (437, {"reason": "target_present_dustbin_reject", "present": True, "predicted": False, "correct": False}),
        (142, {"reason": "target_present_wrong_selection", "present": True, "predicted": True, "correct": False}),
    ):
        for _ in range(count):
            rows.append(audit_row(index=index, **kwargs))
            index += 1
    for reason, count in (
        ("interval_merge", 413),
        ("no_opening_signal_A", 223),
        ("no_opening_signal_B", 63),
        ("no_opening_signal_both", 7),
        ("below_threshold_A", 14),
        ("below_threshold_B", 5),
        ("interval_split", 16),
        ("dedup_removed", 4),
        ("candidate_cap_removed", 1),
        ("interval_expansion_failure", 1),
    ):
        for _ in range(count):
            rows.append(audit_row(reason, index=index))
            index += 1
    return rows


class VisualAuditTests(unittest.TestCase):
    def test_rejects_test_split_path(self):
        with self.assertRaises(ValueError):
            AUDIT.reject_test_data_path("/data/test/manifest.json", "manifest")
        with self.assertRaises(ValueError):
            AUDIT.reject_test_data_path("/data/splits/test.json", "manifest")
        self.assertEqual(AUDIT.reject_test_data_path("/tmp/tests/val.json", "manifest").name, "val.json")

    def test_exact_statistics_gate_and_mismatch(self):
        rows = exact_full_val_rows()
        reproduced = AUDIT.validate_statistics(rows)
        self.assertEqual(reproduced["funnel"], AUDIT.EXPECTED_FUNNEL)
        self.assertEqual(reproduced["missing_reasons"], AUDIT.EXPECTED_MISSING)
        rows[0]["legacy"]["selected_correct"] = False
        with self.assertRaises(RuntimeError):
            AUDIT.validate_statistics(rows)

    def test_house_first_selection_is_permutation_invariant(self):
        candidates = [
            {"pair_id": "p{}".format(index), "house_id": "h{}".format(index // 2)}
            for index in range(12)
        ]
        severity = {row["pair_id"]: float(index) for index, row in enumerate(candidates)}
        first = AUDIT.stratified_house_first(candidates, 4, severity, set())
        second = AUDIT.stratified_house_first(list(reversed(candidates)), 4, severity, set())
        first_ids = [row["pair_id"] for row, _ in first]
        second_ids = [row["pair_id"] for row, _ in second]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len({row["house_id"] for row, _ in first}), 4)
        self.assertEqual(AUDIT.canonical_sha(first_ids), AUDIT.canonical_sha(second_ids))

    def test_wrap_token_to_pixel_spans_has_no_off_by_one(self):
        self.assertEqual(AUDIT.interval_tokens([14, 1], 16), [14, 15, 0, 1])
        self.assertEqual(AUDIT.interval_pixel_spans([14, 1], 16, 64), [(56.0, 64.0), (0.0, 8.0)])
        self.assertEqual(AUDIT.interval_pixel_spans([2, 4], 16, 64), [(8.0, 20.0)])

    def test_dustbin_recovery_metrics_and_sentinel_provenance(self):
        dustbin = 0.5
        score = {
            "candidate_count_A": 2,
            "candidate_count_B": 2,
            "candidate_pair_scores": [[0.1, 0.2], [0.3, 0.4]],
            "candidate_assignment_logits": [[0.0, 0.2], [0.4, 0.6]],
            "row_dustbin_logits": [-3.402823466e38, -3.402823466e38],
            "column_dustbin_logits": [dustbin, dustbin],
            "best_candidate_pair": [1, 1],
            "pair_match_probability": 0.4,
            "pair_no_match_probability": AUDIT.recover_pair_no_match([[0.0, 0.2], [0.4, 0.6]], dustbin),
        }
        row = {"legacy": {"target_pairs": [[0, 1]]}}
        metrics = AUDIT.matcher_metrics(row, score, dustbin)
        self.assertEqual(metrics["true_pair"], [0, 1])
        self.assertAlmostEqual(metrics["true_pair_score"], 0.2)
        self.assertAlmostEqual(metrics["selected_vs_true_score_gap"], 0.2)
        self.assertAlmostEqual(metrics["true_to_dustbin_logit_margin"], -0.3)
        self.assertEqual(metrics["row_dustbin_provenance"], "recovered_exact_from_shared_dustbin_constant")
        self.assertEqual(metrics["column_dustbin_provenance"], "direct")

    def test_observation_review_gate_and_strict_json(self):
        selected = {
            "pair_id": "p1",
            "audit_category": "interval_merge",
            "automatic_reason": "interval_merge",
            "figure_path": "figures/p1.png",
            "figure_sha256": "abc",
        }
        observation = {
            "pair_id": "p1",
            "reviewed": True,
            "reviewed_figure_sha256": "abc",
            "gt_opening_visibility": "clear",
            "viewpoint": "frontal",
            "occlusion": "none",
            "illumination": "normal",
            "nearby_openings": 1,
            "repeated_door_appearance": False,
            "wrap_or_seam": False,
            "annotation_ambiguity": "none",
            "observed_primary_mechanism": "broad response bridge",
            "observed_secondary_mechanism": "none",
            "confidence": "high",
            "automatic_attribution_assessment": "confirmed",
            "visual_evidence": ["A single wide response crosses the highlighted interval."],
            "direct_visual_facts": ["The response is above threshold across the interval."],
            "inferences": [],
            "unresolved": [],
        }
        validated = AUDIT._validate_observations([observation], {"selected": [selected]})
        self.assertEqual(validated[0]["nearby_openings"], "1")
        bad = dict(observation)
        bad["reviewed"] = False
        with self.assertRaises(ValueError):
            AUDIT._validate_observations([bad], {"selected": [selected]})
        with self.assertRaises(ValueError):
            AUDIT.strict_json_text({"bad": float("nan")})
        with self.assertRaises(ValueError):
            AUDIT.strict_json_text({"bad": float("inf")})

    def test_cli_has_no_threshold_or_device_override(self):
        parser_error = None
        try:
            AUDIT.parse_args(["render", "--threshold", "0.2"])
        except SystemExit as exc:
            parser_error = exc
        self.assertIsNotNone(parser_error)
        parser_error = None
        try:
            AUDIT.parse_args(["render", "--device", "cuda:0"])
        except SystemExit as exc:
            parser_error = exc
        self.assertIsNotNone(parser_error)

    def test_tiny_panel_png_smoke_uses_trace_threshold(self):
        import numpy as np
        import torch
        from PIL import Image

        probability = torch.tensor([0.05, 0.1, 0.35, 0.6, 0.5, 0.1, 0.05, 0.02], dtype=torch.float32)
        trace = {
            'portallayout.config': {"threshold": 0.31},
            "threshold_components": [{"interval": [2, 4]}],
            "local_peaks": [{
                "peak_index": 3, "peak_probability": 0.6, "interval": [1, 5],
                "removed_by_dedup": False, "removed_by_cap": False,
                "removed_by_peak_top_k": False,
            }],
        }
        room = {
            "token_count": 8,
            "opening_probability": probability,
            "candidate_trace": trace,
            "legacy_candidates": [[2, 4]],
        }
        target = {
            "target_interval": [2, 4], "target_wrap": False,
            "nearest_peak": {"peak_index": 3},
            "stages": {
                "threshold": {"best": {"interval": [2, 4]}},
                "raw": {"best": {"interval": [1, 5]}},
                "legacy_final": {"recalled": True, "best": {"interval": [2, 4]}},
            },
        }
        row = {
            "pair_id": "synthetic", "primary_pair_reason": "target_present_correct",
            "house_id": "h", "room_A": "ra", "room_B": "rb", "pano_A": "pa", "pano_B": "pb",
            "target_A": target, "target_B": target,
        }
        metrics = {
            "target_pairs": [[0, 0]], "selected_pair": [0, 0],
            "candidate_pair_scores": [[0.8]], "candidate_count_A": 1,
            "candidate_count_B": 1, "candidate_product": 1, "true_pair": [0, 0],
            "true_pair_score": 0.8, "true_pair_raw_logit": 1.0,
            "true_real_real_rank": 1, "selected_pair_score": 0.8,
            "selected_vs_true_score_gap": 0.0, "shared_dustbin_logit": 0.5,
            "true_to_dustbin_logit_margin": 0.5, "row_dustbin_provenance": "direct",
            "column_dustbin_provenance": "direct", "pair_match_probability": 0.8,
            "pair_no_match_probability": 0.2,
        }
        image = np.full((32, 64, 3), 180, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.png"
            AUDIT.render_panel(path, row, {"A": room, "B": room}, {"A": image, "B": image}, metrics)
            with Image.open(str(path)) as rendered:
                self.assertEqual(rendered.format, "PNG")
                self.assertGreater(rendered.width, 1000)
                self.assertGreater(rendered.height, 1000)


if __name__ == "__main__":
    unittest.main()
