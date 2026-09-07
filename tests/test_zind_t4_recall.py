import unittest
from types import SimpleNamespace

import torch

from models.cross_scene_matcher import (
    candidate_assignment_loss,
    candidate_intervals_to_mask,
    candidate_listwise_ranking_loss,
    opening_probabilities_to_intervals,
    positive_to_dustbin_margin_loss,
    recall_topk_opening_intervals,
)
from tools.train_zind_cross_scene_matcher import predicted_candidate_masks
from tools.train_zind_end_to_end import _candidate_inputs


class RecallProposalTest(unittest.TestCase):
    def test_legacy_mode_is_elementwise_compatible(self):
        torch.manual_seed(123)
        probability = torch.rand(3, 32)
        masks, valid = predicted_candidate_masks(
            probability, 0.47, 2, 12, proposal_mode="legacy"
        )
        for index in range(3):
            intervals = opening_probabilities_to_intervals(
                probability[index], 0.47, 2, 12
            )
            expected = candidate_intervals_to_mask(intervals, 32)
            count = len(intervals)
            torch.testing.assert_close(masks[index, :count], expected)
            self.assertEqual(valid[index].tolist(), [True] * count + [False] * (valid.shape[1] - count))

    def test_topk_is_deterministic_wraps_deduplicates_and_caps(self):
        probability = torch.zeros(16)
        probability[15] = 0.90
        probability[0] = 0.85
        probability[5] = 0.80
        probability[10] = 0.70
        first = recall_topk_opening_intervals(
            probability,
            threshold=0.95,
            min_width_tokens=2,
            max_intervals=2,
            peak_top_k=4,
            peak_min_probability=0.60,
            expansion_threshold=0.20,
            dedup_iou=0.50,
        )
        second = recall_topk_opening_intervals(
            probability,
            threshold=0.95,
            min_width_tokens=2,
            max_intervals=2,
            peak_top_k=4,
            peak_min_probability=0.60,
            expansion_threshold=0.20,
            dedup_iou=0.50,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0], (15, 0))

        with_legacy = recall_topk_opening_intervals(
            probability,
            threshold=0.82,
            min_width_tokens=2,
            max_intervals=4,
            peak_top_k=4,
            peak_min_probability=0.60,
            expansion_threshold=0.20,
            dedup_iou=0.50,
        )
        self.assertEqual(sum(interval == (15, 0) for interval in with_legacy), 1)

    def test_batch_padding_is_finite_and_reproducible(self):
        probability = torch.tensor(
            [[0.9, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        first = predicted_candidate_masks(
            probability,
            0.7,
            2,
            3,
            proposal_mode="recall_topk",
            peak_top_k=1,
            peak_min_probability=0.5,
            expansion_threshold=0.2,
        )
        second = predicted_candidate_masks(
            probability,
            0.7,
            2,
            3,
            proposal_mode="recall_topk",
            peak_top_k=1,
            peak_min_probability=0.5,
            expansion_threshold=0.2,
        )
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertTrue(bool(torch.isfinite(first[0].float()).all()))


class _FixedOpeningHead:
    def __call__(self, features, enclosed, extended):
        return {"opening_probability": torch.full(features.shape[:2], 0.10, device=features.device)}


class _CandidateModel:
    def __init__(self, training):
        self.training = training
        self.matcher = SimpleNamespace(opening_head=_FixedOpeningHead())


class TrainInjectionIsolationTest(unittest.TestCase):
    def _inputs(self, training):
        live = {
            "layout_feature_A": torch.zeros(1, 8, 2),
            "layout_feature_B": torch.zeros(1, 8, 2),
            "enclosed_depth_A": torch.ones(1, 8),
            "extended_depth_A": torch.ones(1, 8),
            "enclosed_depth_B": torch.ones(1, 8),
            "extended_depth_B": torch.ones(1, 8),
        }
        target_a = candidate_intervals_to_mask([(2, 3)], 8).unsqueeze(0)
        target_b = candidate_intervals_to_mask([(5, 6)], 8).unsqueeze(0)
        affinity = torch.outer(target_a[0, 0], target_b[0, 0]).float().unsqueeze(0)
        raw = {
            "manifest_candidate_masks_A": target_a,
            "manifest_candidate_masks_B": target_b,
            "candidate_target_AB": torch.ones(1, 1, 1),
        }
        pair = SimpleNamespace(affinity_ab=affinity)
        args = SimpleNamespace(
            min_opening_width_tokens=2,
            max_openings_per_view=4,
            candidate_proposal_mode="legacy",
            proposal_peak_top_k=4,
            proposal_peak_min_probability=0.2,
            proposal_expansion_threshold=0.12,
            proposal_dedup_iou=0.5,
            train_gt_candidate_injection=True,
        )
        result = _candidate_inputs(
            _CandidateModel(training), live, raw, pair, "predicted", 0.47, args
        )
        return result, raw

    def test_train_injects_but_validation_is_predicted_only(self):
        train, train_raw = self._inputs(True)
        validation, validation_raw = self._inputs(False)
        self.assertGreater(int(train_raw["_t4_gt_candidate_injection_count"]), 0)
        self.assertTrue(bool(train[4].any()))
        self.assertEqual(int(validation_raw["_t4_gt_candidate_injection_count"]), 0)
        self.assertFalse(bool(validation[4].any()))


def _margin_outputs(logits):
    batch, count_a_plus, count_b_plus = logits.shape
    count_a, count_b = count_a_plus - 1, count_b_plus - 1
    valid_a = torch.ones(batch, count_a, dtype=torch.bool)
    valid_b = torch.ones(batch, count_b, dtype=torch.bool)
    return {
        "candidate_assignment_logits": logits,
        "candidate_assignment_AB": torch.softmax(logits, dim=-1),
        "candidate_assignment_BA": torch.softmax(logits.transpose(-2, -1), dim=-1),
        "candidate_valid_A": valid_a,
        "candidate_valid_B": valid_b,
    }, valid_a, valid_b


class DustbinMarginTest(unittest.TestCase):
    def test_positive_target_acts_but_missing_and_negative_are_masked(self):
        logits = torch.zeros(1, 2, 2, requires_grad=True)
        outputs, valid_a, valid_b = _margin_outputs(logits)
        target = torch.ones(1, 1, 1)
        loss = positive_to_dustbin_margin_loss(
            outputs, target, torch.tensor([True]), valid_a, valid_b, margin=0.2
        )
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertLess(float(logits.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 1]), 0.0)

        for target_value, match in ((torch.zeros_like(target), True), (target, False)):
            local = torch.zeros(1, 2, 2, requires_grad=True)
            local_outputs, va, vb = _margin_outputs(local)
            masked = positive_to_dustbin_margin_loss(
                local_outputs, target_value, torch.tensor([match]), va, vb, margin=0.2
            )
            masked.backward()
            self.assertEqual(float(masked.item()), 0.0)
            torch.testing.assert_close(local.grad, torch.zeros_like(local.grad))

    def test_margin_is_learnable_and_negative_assignment_is_unchanged(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)

        def build():
            real, dust_a, dust_b = parameter
            logits = torch.stack((real, dust_a, dust_b, parameter.new_tensor(0.0))).reshape(1, 2, 2)
            return _margin_outputs(logits)

        outputs, va, vb = build()
        initial = positive_to_dustbin_margin_loss(
            outputs, torch.ones(1, 1, 1), torch.tensor([True]), va, vb, margin=0.5
        )
        initial_value = float(initial.item())
        for _ in range(20):
            optimizer.zero_grad()
            outputs, va, vb = build()
            loss = positive_to_dustbin_margin_loss(
                outputs, torch.ones(1, 1, 1), torch.tensor([True]), va, vb, margin=0.5
            )
            loss.backward()
            optimizer.step()
        outputs, va, vb = build()
        final = positive_to_dustbin_margin_loss(
            outputs, torch.ones(1, 1, 1), torch.tensor([True]), va, vb, margin=0.5
        )
        self.assertLess(float(final.item()), initial_value * 0.2)
        self.assertGreater(float(parameter[0] - parameter[1]), 0.45)

        negative_logits = torch.zeros(1, 2, 2, requires_grad=True)
        negative_outputs, va, vb = _margin_outputs(negative_logits)
        before = candidate_assignment_loss(
            negative_outputs,
            candidate_target_ab=torch.zeros(1, 1, 1),
            is_match=torch.tensor([False]),
            consistency_weight=0.0,
        )["loss_candidate_total"]
        margin = positive_to_dustbin_margin_loss(
            negative_outputs,
            torch.zeros(1, 1, 1),
            torch.tensor([False]),
            va,
            vb,
        )
        torch.testing.assert_close(before + 0.3 * margin, before, rtol=0.0, atol=0.0)

    def test_ranking_loss_is_unchanged_and_padding_stays_finite(self):
        affinity = torch.tensor([[[0.8, 0.0], [0.0, 0.0]]])
        confidence = torch.ones_like(affinity)
        valid = torch.tensor([[True, False]])
        ranking_outputs = {
            "candidate_mutual_affinity": affinity,
            "candidate_opening_confidence": confidence,
            "candidate_valid_A": valid,
            "candidate_valid_B": valid,
        }
        target = torch.zeros(1, 2, 2)
        target[0, 0, 0] = 1.0
        before = candidate_listwise_ranking_loss(
            ranking_outputs, target, torch.tensor([True]), valid, valid
        )
        logits = torch.tensor([[[0.0, -torch.inf, 0.0], [-torch.inf, -torch.inf, -torch.inf], [0.0, -torch.inf, 0.0]]])
        outputs = {"candidate_assignment_logits": logits, "candidate_valid_A": valid, "candidate_valid_B": valid}
        margin = positive_to_dustbin_margin_loss(
            outputs, target, torch.tensor([True]), valid, valid
        )
        after = candidate_listwise_ranking_loss(
            ranking_outputs, target, torch.tensor([True]), valid, valid
        )
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
        self.assertTrue(bool(torch.isfinite(margin)))

        padded_no_target = positive_to_dustbin_margin_loss(
            outputs,
            torch.zeros_like(target),
            torch.tensor([False]),
            valid,
            valid,
        )
        self.assertEqual(float(padded_no_target.item()), 0.0)
        self.assertTrue(bool(torch.isfinite(padded_no_target)))


if __name__ == "__main__":
    unittest.main()
