import json
import unittest

import torch

from models.cross_scene_matcher import (
    circular_interval_iou,
    opening_probabilities_to_intervals,
    recall_topk_opening_intervals,
)
from tools.audit_zind_candidate_failures import (
    _bootstrap_decision_categories,
    _decision_categories,
    _physical_matches,
    _target_pairs,
)
from utils.candidate_failure_trace import (
    assign_side_reason,
    combine_side_reason,
    finite_tree,
    target_interval_diagnostics,
    trace_opening_proposals,
)


class CandidateFailureTraceTests(unittest.TestCase):
    def test_circular_iou_and_wrap(self):
        self.assertAlmostEqual(circular_interval_iou((14, 1), (15, 2), 16), 3 / 5)
        probability = torch.zeros(16)
        probability[[14, 15, 0, 1]] = 0.9
        trace = trace_opening_proposals(probability, threshold=0.47)
        self.assertEqual(trace["legacy_candidates"], [[14, 1]])
        self.assertTrue(trace["threshold_components"][0]["wrap"])

    def test_trace_exactly_preserves_production_outputs(self):
        generator = torch.Generator().manual_seed(123)
        probability = torch.rand(256, generator=generator)
        trace = trace_opening_proposals(probability)
        legacy = opening_probabilities_to_intervals(
            probability, threshold=0.47, min_width_tokens=2, max_intervals=12
        )
        recall = recall_topk_opening_intervals(
            probability, threshold=0.47, min_width_tokens=2, max_intervals=12,
            peak_top_k=4, peak_min_probability=0.20,
            expansion_threshold=0.12, dedup_iou=0.50,
        )
        self.assertEqual(trace["legacy_candidates"], [list(value) for value in legacy])
        self.assertEqual(trace["recall_topk_candidates"], [list(value) for value in recall])
        self.assertFalse(trace["gt_available_to_production"])

    def test_threshold_and_no_signal_reasons(self):
        below = torch.zeros(32)
        below[8:12] = 0.30
        trace = trace_opening_proposals(below)
        diagnostic = target_interval_diagnostics(below, trace, (8, 11))
        self.assertEqual(assign_side_reason(diagnostic, trace), "below_threshold")
        absent = torch.zeros(32)
        trace = trace_opening_proposals(absent)
        diagnostic = target_interval_diagnostics(absent, trace, (8, 11))
        self.assertEqual(assign_side_reason(diagnostic, trace), "no_opening_signal")
        self.assertEqual(combine_side_reason("no_opening_signal", "present"), "no_opening_signal_A")
        self.assertEqual(combine_side_reason("present", "no_opening_signal"), "no_opening_signal_B")
        self.assertEqual(combine_side_reason("no_opening_signal", "no_opening_signal"), "no_opening_signal_both")

    def test_split_merge_dedup_and_cap_are_traced(self):
        split = torch.zeros(32)
        split[7:9] = 0.9
        split[11:13] = 0.9
        trace = trace_opening_proposals(split)
        diagnostic = target_interval_diagnostics(split, trace, (7, 15))
        self.assertEqual(assign_side_reason(diagnostic, trace), "interval_split")

        merged = torch.zeros(32)
        merged[2:20] = 0.9
        trace = trace_opening_proposals(merged)
        diagnostic = target_interval_diagnostics(merged, trace, (8, 10))
        self.assertEqual(assign_side_reason(diagnostic, trace), "interval_merge")

        signal = torch.tensor([0.1, 0.7, 0.7, 0.1, 0.6, 0.1, 0.5, 0.1])
        capped = trace_opening_proposals(signal, max_candidates=1, peak_top_k=4)
        deduped = trace_opening_proposals(signal, max_candidates=12, peak_top_k=4)
        self.assertTrue(any(row["removed_by_cap"] for row in capped["local_peaks"]))
        self.assertTrue(any(row["removed_by_dedup"] for row in deduped["local_peaks"]))

    def test_target_mapping_multi_shared_and_window_exclusion(self):
        def candidate(kind, interval, y):
            return {
                "type": kind, "tokenInterval": interval,
                "globalEndpoints": [[0.0, y], [1.0, y]],
            }

        record = {
            "token_count": 16,
            "candidates_a": [candidate("door", [1, 3], 0.0), candidate("opening", [8, 10], 1.0), candidate("window", [12, 13], 2.0)],
            "candidates_b": [candidate("door", [4, 6], 0.0), candidate("opening", [11, 13], 1.0), candidate("window", [14, 15], 2.0)],
        }
        self.assertEqual(_physical_matches(record), [(0, 0), (1, 1)])
        targets = _target_pairs(record, [(1, 3), (8, 10)], [(4, 6), (11, 13)])
        self.assertEqual(targets, [[0, 0], [1, 1]])

    def test_deterministic_reason_and_json_finite(self):
        probability = torch.linspace(0, 1, 32)
        first = trace_opening_proposals(probability)
        second = trace_opening_proposals(probability)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertTrue(finite_tree(first))
        self.assertFalse(finite_tree({"bad": float("nan")}))

    def test_house_bootstrap_covers_all_failure_categories(self):
        reasons = [
            ("h1", "no_opening_signal_A"),
            ("h1", "interval_merge"),
            ("h2", "side_candidates_present_pair_target_missing"),
            ("h2", "target_present_dustbin_reject"),
            ("h3", "target_present_correct"),
        ]
        rows = [{"house_id": house, "primary_pair_reason": reason} for house, reason in reasons]
        self.assertEqual(
            _decision_categories(rows),
            {
                "A_opening_head_signal": 1, "B_proposal_logic": 1,
                "C_target_mapping": 1, "D_matcher": 1, "success": 1,
            },
        )
        first = _bootstrap_decision_categories(rows, replicates=100, seed=123)
        second = _bootstrap_decision_categories(rows, replicates=100, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["failure_pair_count"], 4)
        self.assertEqual(first["success_pair_count"], 1)
        self.assertEqual(sum(v["count"] for v in first["categories"].values()), 4)


if __name__ == "__main__":
    unittest.main()
