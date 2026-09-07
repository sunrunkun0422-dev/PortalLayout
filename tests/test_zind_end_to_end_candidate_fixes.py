import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.cross_scene_matcher import (
    OpeningGuidedCrossAttentionMatcher,
    candidate_assignment_loss,
    candidate_listwise_ranking_loss,
    candidate_pair_score,
)
from tools.train_zind_end_to_end import (
    _active_splits,
    _build_evaluation_loaders,
    _build_split_datasets,
    _candidate_inputs,
    _candidate_supervision_validity,
    _data_contract,
    _opening_guidance_inputs,
    _resolve_split_paths,
    _resolve_source_configuration,
    _schedule,
    _validate_resume_contract,
    parse_args,
)


def _assignment_outputs(
    logits_ab: torch.Tensor,
    logits_ba: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
):
    return {
        "candidate_assignment_AB": torch.softmax(logits_ab, dim=-1),
        "candidate_assignment_BA": torch.softmax(logits_ba, dim=-1),
        "candidate_valid_A": valid_a,
        "candidate_valid_B": valid_b,
    }


def _ranking_outputs(
    affinity: torch.Tensor,
    confidence: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
):
    return {
        "candidate_mutual_affinity": affinity,
        "candidate_opening_confidence": confidence,
        "candidate_valid_A": valid_a,
        "candidate_valid_B": valid_b,
        "candidate_pair_score": candidate_pair_score(
            affinity, confidence, valid_a, valid_b
        ),
    }


class CandidateAssignmentMaskTest(unittest.TestCase):
    def test_negative_without_target_keeps_dustbin_loss_and_gradient(self):
        logits_ab = torch.zeros(1, 2, 2, requires_grad=True)
        logits_ba = torch.zeros(1, 2, 2, requires_grad=True)
        valid = torch.ones(1, 1, dtype=torch.bool)
        outputs = _assignment_outputs(logits_ab, logits_ba, valid, valid)
        sample_valid, has_target = _candidate_supervision_validity(
            torch.zeros(1, 1, 1),
            torch.tensor([False]),
            valid,
            valid,
            "mask_positive",
        )

        loss = candidate_assignment_loss(
            outputs,
            candidate_target_ab=torch.zeros(1, 1, 1),
            is_match=torch.tensor([False]),
            sample_valid=sample_valid,
            consistency_weight=0.0,
        )["loss_candidate_total"]
        loss.backward()

        self.assertFalse(bool(has_target.item()))
        self.assertTrue(bool(sample_valid.item()))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertLess(float(logits_ab.grad[0, 0, 1].item()), 0.0)
        self.assertLess(float(logits_ba.grad[0, 0, 1].item()), 0.0)

    def test_positive_without_target_is_fully_masked(self):
        logits_ab = torch.zeros(1, 2, 2, requires_grad=True)
        logits_ba = torch.zeros(1, 2, 2, requires_grad=True)
        valid = torch.ones(1, 1, dtype=torch.bool)
        outputs = _assignment_outputs(logits_ab, logits_ba, valid, valid)
        sample_valid, _ = _candidate_supervision_validity(
            torch.zeros(1, 1, 1),
            torch.tensor([True]),
            valid,
            valid,
            "mask_positive",
        )

        loss = candidate_assignment_loss(
            outputs,
            candidate_target_ab=torch.zeros(1, 1, 1),
            is_match=torch.tensor([True]),
            sample_valid=sample_valid,
            consistency_weight=0.0,
        )["loss_candidate_total"]
        loss.backward()

        self.assertFalse(bool(sample_valid.item()))
        self.assertEqual(float(loss.item()), 0.0)
        torch.testing.assert_close(logits_ab.grad, torch.zeros_like(logits_ab))
        torch.testing.assert_close(logits_ba.grad, torch.zeros_like(logits_ba))

    def test_positive_target_keeps_unmatched_rows_on_dustbin(self):
        logits_ab = torch.zeros(1, 3, 3, requires_grad=True)
        logits_ba = torch.zeros(1, 3, 3, requires_grad=True)
        valid = torch.ones(1, 2, dtype=torch.bool)
        target = torch.zeros(1, 2, 2)
        target[0, 0, 0] = 1.0
        outputs = _assignment_outputs(logits_ab, logits_ba, valid, valid)

        loss = candidate_assignment_loss(
            outputs,
            candidate_target_ab=target,
            is_match=torch.tensor([True]),
            sample_valid=torch.tensor([True]),
            consistency_weight=0.0,
        )["loss_candidate_total"]
        loss.backward()

        self.assertLess(float(logits_ab.grad[0, 0, 0].item()), 0.0)
        self.assertLess(float(logits_ab.grad[0, 1, 2].item()), 0.0)
        self.assertLess(float(logits_ba.grad[0, 1, 2].item()), 0.0)

    def test_mixed_batch_uses_only_valid_samples(self):
        logits_ab = torch.tensor(
            [
                [[1.5, -0.5], [-0.5, 1.5]],
                [[-0.3, 0.7], [0.7, -0.3]],
            ],
            requires_grad=True,
        )
        logits_ba = logits_ab.detach().clone().requires_grad_(True)
        valid = torch.ones(2, 1, dtype=torch.bool)
        target = torch.zeros(2, 1, 1)
        is_match = torch.tensor([True, False])
        sample_valid, _ = _candidate_supervision_validity(
            target, is_match, valid, valid, "mask_positive"
        )
        outputs = _assignment_outputs(logits_ab, logits_ba, valid, valid)
        mixed = candidate_assignment_loss(
            outputs,
            candidate_target_ab=target,
            is_match=is_match,
            sample_valid=sample_valid,
            consistency_weight=0.0,
        )["loss_candidate_total"]

        isolated_outputs = {
            "candidate_assignment_AB": outputs["candidate_assignment_AB"][1:],
            "candidate_assignment_BA": outputs["candidate_assignment_BA"][1:],
            "candidate_valid_A": valid[1:],
            "candidate_valid_B": valid[1:],
        }
        isolated = candidate_assignment_loss(
            isolated_outputs,
            candidate_target_ab=target[1:],
            is_match=is_match[1:],
            consistency_weight=0.0,
        )["loss_candidate_total"]

        self.assertEqual(sample_valid.tolist(), [False, True])
        torch.testing.assert_close(mixed, isolated)

    def test_all_samples_masked_returns_graph_connected_zero(self):
        logits_ab = torch.randn(2, 2, 2, requires_grad=True)
        logits_ba = torch.randn(2, 2, 2, requires_grad=True)
        valid = torch.ones(2, 1, dtype=torch.bool)
        outputs = _assignment_outputs(logits_ab, logits_ba, valid, valid)

        loss = candidate_assignment_loss(
            outputs,
            candidate_target_ab=torch.zeros(2, 1, 1),
            is_match=torch.ones(2, dtype=torch.bool),
            sample_valid=torch.zeros(2, dtype=torch.bool),
            consistency_weight=0.1,
        )["loss_candidate_total"]
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(loss.item()), 0.0)
        self.assertIsNotNone(logits_ab.grad)
        self.assertIsNotNone(logits_ba.grad)

    def test_all_valid_matches_legacy_loss_exactly(self):
        torch.manual_seed(101)
        logits_ab = torch.randn(2, 3, 3)
        logits_ba = torch.randn(2, 3, 3)
        valid = torch.tensor([[True, True], [True, False]])
        outputs = _assignment_outputs(logits_ab, logits_ba, valid, valid)
        target = torch.zeros(2, 2, 2)
        target[0, 0, 1] = 1.0
        target[1, 0, 0] = 1.0
        is_match = torch.ones(2, dtype=torch.bool)

        legacy = candidate_assignment_loss(
            outputs,
            candidate_target_ab=target,
            is_match=is_match,
            consistency_weight=0.0,
        )["loss_candidate_total"]
        explicit_all_valid = candidate_assignment_loss(
            outputs,
            candidate_target_ab=target,
            is_match=is_match,
            sample_valid=torch.ones(2, dtype=torch.bool),
            consistency_weight=0.0,
        )["loss_candidate_total"]

        torch.testing.assert_close(legacy, explicit_all_valid, rtol=0.0, atol=0.0)

    def test_padding_cannot_create_a_target(self):
        target = torch.zeros(1, 2, 2)
        target[0, 1, 1] = 1.0
        valid = torch.tensor([[True, False]])
        sample_valid, has_target = _candidate_supervision_validity(
            target,
            torch.tensor([True]),
            valid,
            valid,
            "mask_positive",
        )

        self.assertFalse(bool(has_target.item()))
        self.assertFalse(bool(sample_valid.item()))


class CandidateRankingLossTest(unittest.TestCase):
    def test_product_and_log_scores_have_identical_order(self):
        affinity = torch.tensor([[[0.2, 0.8], [0.5, 0.1]]])
        confidence = torch.tensor([[[0.9, 0.3], [0.7, 0.4]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        product = candidate_pair_score(affinity, confidence, valid, valid)
        legacy_product = affinity * confidence * (
            valid.unsqueeze(-1) & valid.unsqueeze(1)
        ).to(affinity.dtype)
        log_score = torch.log(product + 1e-8)

        torch.testing.assert_close(product, legacy_product, rtol=0.0, atol=0.0)
        self.assertEqual(
            torch.argsort(product.flatten(), descending=True).tolist(),
            torch.argsort(log_score.flatten(), descending=True).tolist(),
        )

    def test_raising_positive_score_lowers_ranking_loss(self):
        valid_a = torch.ones(1, 1, dtype=torch.bool)
        valid_b = torch.ones(1, 2, dtype=torch.bool)
        confidence = torch.ones(1, 1, 2)
        target = torch.tensor([[[1.0, 0.0]]])
        low = _ranking_outputs(
            torch.tensor([[[0.2, 0.8]]]), confidence, valid_a, valid_b
        )
        high = _ranking_outputs(
            torch.tensor([[[0.9, 0.8]]]), confidence, valid_a, valid_b
        )

        low_loss = candidate_listwise_ranking_loss(
            low, target, torch.tensor([True])
        )
        high_loss = candidate_listwise_ranking_loss(
            high, target, torch.tensor([True])
        )

        self.assertLess(float(high_loss.item()), float(low_loss.item()))

    def test_multiple_positives_accumulate_probability_mass(self):
        valid_a = torch.ones(1, 1, dtype=torch.bool)
        valid_b = torch.ones(1, 3, dtype=torch.bool)
        outputs = _ranking_outputs(
            torch.tensor([[[0.2, 0.3, 0.5]]]),
            torch.ones(1, 1, 3),
            valid_a,
            valid_b,
        )
        one_positive = torch.tensor([[[1.0, 0.0, 0.0]]])
        two_positive = torch.tensor([[[1.0, 1.0, 0.0]]])

        one_loss = candidate_listwise_ranking_loss(
            outputs, one_positive, torch.tensor([True])
        )
        two_loss = candidate_listwise_ranking_loss(
            outputs, two_positive, torch.tensor([True])
        )

        self.assertLess(float(two_loss.item()), float(one_loss.item()))

    def test_padded_candidate_does_not_change_loss(self):
        valid_a = torch.ones(1, 1, dtype=torch.bool)
        valid_b = torch.tensor([[True, False]])
        target = torch.tensor([[[1.0, 0.0]]])
        confidence = torch.ones(1, 1, 2)
        normal = _ranking_outputs(
            torch.tensor([[[0.4, 1.0]]]), confidence, valid_a, valid_b
        )
        extreme_padding = _ranking_outputs(
            torch.tensor([[[0.4, 1e6]]]), confidence, valid_a, valid_b
        )

        normal_loss = candidate_listwise_ranking_loss(
            normal, target, torch.tensor([True])
        )
        padded_loss = candidate_listwise_ranking_loss(
            extreme_padding, target, torch.tensor([True])
        )

        torch.testing.assert_close(normal_loss, padded_loss)

    def test_missing_target_returns_graph_connected_zero(self):
        affinity = torch.tensor([[[0.4, 0.6]]], requires_grad=True)
        valid_a = torch.ones(1, 1, dtype=torch.bool)
        valid_b = torch.ones(1, 2, dtype=torch.bool)
        outputs = _ranking_outputs(
            affinity, torch.ones(1, 1, 2), valid_a, valid_b
        )

        loss = candidate_listwise_ranking_loss(
            outputs, torch.zeros(1, 1, 2), torch.tensor([True])
        )
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(loss.item()), 0.0)
        torch.testing.assert_close(affinity.grad, torch.zeros_like(affinity))

    def test_matcher_inference_and_shared_training_score_rank_identically(self):
        torch.manual_seed(303)
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=8, heads=2, hidden_dim=4, dropout=0.0
        )
        features_a = torch.randn(1, 6, 8)
        features_b = torch.randn(1, 6, 8)
        enclosed = torch.ones(1, 6)
        extended = enclosed.clone()
        extended[:, 1:5] = 2.0
        masks_a = torch.zeros(1, 2, 6, dtype=torch.bool)
        masks_b = torch.zeros(1, 2, 6, dtype=torch.bool)
        masks_a[0, 0, 0:2] = True
        masks_a[0, 1, 3:5] = True
        masks_b[0, 0, 1:3] = True
        masks_b[0, 1, 4:6] = True
        valid = torch.ones(1, 2, dtype=torch.bool)

        outputs = matcher(
            features_a,
            features_b,
            enclosed,
            extended,
            enclosed,
            extended,
            candidate_masks_a=masks_a,
            candidate_masks_b=masks_b,
            candidate_valid_a=valid,
            candidate_valid_b=valid,
        )
        shared_score = candidate_pair_score(
            outputs["candidate_mutual_affinity"],
            outputs["candidate_opening_confidence"],
            valid,
            valid,
        )

        torch.testing.assert_close(outputs["candidate_pair_score"], shared_score)
        self.assertEqual(
            outputs["candidate_pair_score"].flatten().argsort().tolist(),
            shared_score.flatten().argsort().tolist(),
        )

    def test_padded_zero_candidates_keep_ranking_gradients_finite(self):
        torch.manual_seed(307)
        matcher = OpeningGuidedCrossAttentionMatcher(
            feature_dim=8, heads=2, hidden_dim=4, dropout=0.0
        )
        features_a = torch.randn(1, 6, 8, requires_grad=True)
        features_b = torch.randn(1, 6, 8, requires_grad=True)
        enclosed = torch.ones(1, 6)
        extended = enclosed.clone()
        extended[:, 1:5] = 2.0
        masks_a = torch.zeros(1, 3, 6, dtype=torch.bool)
        masks_b = torch.zeros(1, 3, 6, dtype=torch.bool)
        masks_a[0, 0, 0:2] = True
        masks_a[0, 1, 3:5] = True
        masks_b[0, 0, 1:3] = True
        masks_b[0, 1, 4:6] = True
        valid = torch.tensor([[True, True, False]])

        outputs = matcher(
            features_a,
            features_b,
            enclosed,
            extended,
            enclosed,
            extended,
            candidate_masks_a=masks_a,
            candidate_masks_b=masks_b,
            candidate_valid_a=valid,
            candidate_valid_b=valid,
        )
        target = torch.zeros(1, 3, 3)
        target[0, 0, 0] = 1.0
        loss = candidate_listwise_ranking_loss(
            outputs,
            target,
            torch.tensor([True]),
        )
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(bool(torch.isfinite(features_a.grad).all()))
        self.assertTrue(bool(torch.isfinite(features_b.grad).all()))
        parameter_gradients = [
            parameter.grad
            for parameter in matcher.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(parameter_gradients)
        self.assertTrue(
            all(bool(torch.isfinite(gradient).all()) for gradient in parameter_gradients)
        )
        self.assertTrue(
            any(bool((gradient.abs() > 0).any()) for gradient in parameter_gradients)
        )


class _FakeOpeningHead:
    def __call__(self, features, enclosed, extended):
        probability = features.new_tensor([[0.9, 0.9, 0.1, 0.1]])
        return {"opening_probability": probability.expand(features.shape[0], -1)}


class _FakeModel:
    def __init__(self):
        self.matcher = SimpleNamespace(opening_head=_FakeOpeningHead())


class _NoManifestCandidateReads(dict):
    def __getitem__(self, key):
        if key.startswith("manifest_candidate") or key == "candidate_target_AB":
            raise AssertionError("predicted candidates read GT candidate data")
        return super().__getitem__(key)


class TeacherForcingSourceTest(unittest.TestCase):
    def setUp(self):
        self.live = {
            "layout_feature_A": torch.zeros(1, 4, 2),
            "layout_feature_B": torch.ones(1, 4, 2),
            "enclosed_depth_A": torch.ones(1, 4),
            "extended_depth_A": torch.ones(1, 4),
            "enclosed_depth_B": torch.ones(1, 4),
            "extended_depth_B": torch.ones(1, 4),
        }
        gt_mask_a = torch.tensor([[[False, False, True, True]]])
        gt_mask_b = torch.tensor([[[True, False, False, True]]])
        self.raw_batch = {
            "manifest_candidate_masks_A": gt_mask_a,
            "manifest_candidate_masks_B": gt_mask_b,
            "manifest_candidate_valid_A": torch.ones(1, 1, dtype=torch.bool),
            "manifest_candidate_valid_B": torch.ones(1, 1, dtype=torch.bool),
            "candidate_target_AB": torch.ones(1, 1, 1),
        }
        self.pair = SimpleNamespace(
            opening_all_a=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            opening_all_b=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            affinity_ab=torch.zeros(1, 4, 4),
        )
        self.args = SimpleNamespace(
            min_opening_width_tokens=2,
            max_openings_per_view=4,
        )

    def _candidates(self, source, raw_batch=None):
        return _candidate_inputs(
            _FakeModel(),
            self.live,
            self.raw_batch if raw_batch is None else raw_batch,
            self.pair,
            source,
            0.5,
            self.args,
        )

    def test_all_four_source_combinations_are_independent(self):
        candidates = {source: self._candidates(source) for source in ("gt", "predicted")}
        guidance = {
            source: _opening_guidance_inputs(self.pair, source)
            for source in ("gt", "predicted")
        }

        for guidance_source in ("gt", "predicted"):
            for candidate_source in ("gt", "predicted"):
                selected_guidance = guidance[guidance_source]
                selected_candidates = candidates[candidate_source]
                if guidance_source == "gt":
                    torch.testing.assert_close(
                        selected_guidance[0], self.pair.opening_all_a
                    )
                else:
                    self.assertEqual(selected_guidance, (None, None))
                if candidate_source == "gt":
                    torch.testing.assert_close(
                        selected_candidates[0],
                        self.raw_batch["manifest_candidate_masks_A"],
                    )
                else:
                    self.assertEqual(
                        torch.where(selected_candidates[0][0, 0])[0].tolist(),
                        [0, 1],
                    )

        torch.testing.assert_close(candidates["gt"][0], candidates["gt"][0])
        torch.testing.assert_close(
            candidates["predicted"][0], candidates["predicted"][0]
        )

    def test_predicted_predicted_does_not_read_gt_guidance_or_candidates(self):
        class PairWithoutGuidance:
            affinity_ab = torch.zeros(1, 4, 4)

            @property
            def opening_all_a(self):
                raise AssertionError("predicted guidance read GT guidance A")

            @property
            def opening_all_b(self):
                raise AssertionError("predicted guidance read GT guidance B")

        pair = PairWithoutGuidance()
        self.assertEqual(_opening_guidance_inputs(pair, "predicted"), (None, None))
        candidates = _candidate_inputs(
            _FakeModel(),
            self.live,
            _NoManifestCandidateReads(),
            pair,
            "predicted",
            0.5,
            self.args,
        )
        self.assertEqual(torch.where(candidates[0][0, 0])[0].tolist(), [0, 1])

    def test_source_overrides_do_not_modify_each_other(self):
        args = SimpleNamespace(
            gt_guidance_epochs=1,
            mix_guidance_epochs=0,
            opening_guidance_source="predicted",
            candidate_source=None,
        )
        self.assertEqual(_schedule(0, args), ("predicted", "gt"))
        args.opening_guidance_source = None
        args.candidate_source = "predicted"
        self.assertEqual(_schedule(0, args), ("gt", "predicted"))

    def test_legacy_flag_maps_to_gt_gt_and_conflicts_fail(self):
        defaults = parse_args([])
        self.assertEqual(defaults.candidate_missing_target_policy, "legacy_dustbin")
        self.assertEqual(defaults.candidate_rank_loss_weight, 0.0)

        legacy = parse_args(["--teacher_forced_manifest_candidates"])
        self.assertEqual(_resolve_source_configuration(legacy), ("gt", "gt"))

        conflict = parse_args(
            [
                "--teacher_forced_manifest_candidates",
                "--opening_guidance_source",
                "predicted",
            ]
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _resolve_source_configuration(conflict)


class _SplitDatasetStub:
    def __init__(self, split, length=3):
        self.split = split
        self.base = SimpleNamespace(split=split)
        self.token_count = 256
        self.feature_dim = 512
        self._length = length

    def __len__(self):
        return self._length


class SkipTestSplitLifecycleTest(unittest.TestCase):
    @staticmethod
    def _dataset_factory(*args, **kwargs):
        return _SplitDatasetStub(kwargs["expected_split"])

    def test_skip_test_never_resolves_or_constructs_test_split(self):
        class MustNotBeAccessed:
            def __bool__(self):
                raise AssertionError("test split argument was accessed")

            def __fspath__(self):
                raise AssertionError("test split path was resolved")

        args = parse_args(
            [
                "--skip_test",
                "--manifest_root",
                "/tmp/manifests",
                "--cache_root",
                "/tmp/cache",
            ]
        )
        args.test_manifest = MustNotBeAccessed()
        args.test_cache_dir = MustNotBeAccessed()
        active = _active_splits(args)
        manifests, caches = _resolve_split_paths(args, active)

        self.assertEqual(active, ("train", "val"))
        self.assertEqual(set(manifests), {"train", "val"})
        self.assertEqual(set(caches), {"train", "val"})
        with patch(
            "tools.train_zind_end_to_end.JointZInDMatchingDataset",
            side_effect=self._dataset_factory,
        ) as dataset_class:
            datasets = _build_split_datasets(
                args, manifests, caches, active
            )
        self.assertEqual(dataset_class.call_count, 2)
        self.assertEqual(set(datasets), {"train", "val"})

        with patch(
            "tools.train_zind_end_to_end._make_loader",
            side_effect=lambda dataset, *unused, **unused_kwargs: dataset,
        ) as make_loader:
            loaders = _build_evaluation_loaders(
                datasets, args, active, torch.device("cpu")
            )
        self.assertEqual(make_loader.call_count, 1)
        self.assertEqual(set(loaders), {"val"})

        contract = _data_contract(
            manifests,
            caches,
            datasets,
            {"train": 1, "val": 1},
            256,
            512,
            active,
        )
        self.assertFalse(contract["test_evaluation_enabled"])
        self.assertEqual(contract["active_splits"], ["train", "val"])
        self.assertNotIn("test", contract["manifests"])
        self.assertNotIn("test", contract["feature_caches"])
        self.assertNotIn("test", contract["pair_counts"])

    def test_default_behavior_constructs_train_val_and_test(self):
        args = parse_args(
            [
                "--manifest_root",
                "/tmp/manifests",
                "--cache_root",
                "/tmp/cache",
            ]
        )
        active = _active_splits(args)
        manifests, caches = _resolve_split_paths(args, active)
        with patch(
            "tools.train_zind_end_to_end.JointZInDMatchingDataset",
            side_effect=self._dataset_factory,
        ) as dataset_class:
            datasets = _build_split_datasets(
                args, manifests, caches, active
            )
        self.assertEqual(active, ("train", "val", "test"))
        self.assertEqual(dataset_class.call_count, 3)
        self.assertEqual(set(datasets), {"train", "val", "test"})

        with patch(
            "tools.train_zind_end_to_end._make_loader",
            side_effect=lambda dataset, *unused, **unused_kwargs: dataset,
        ) as make_loader:
            loaders = _build_evaluation_loaders(
                datasets, args, active, torch.device("cpu")
            )
        self.assertEqual(make_loader.call_count, 2)
        self.assertEqual(set(loaders), {"val", "test"})

    def test_resume_validates_active_split_contract(self):
        model_contract = {"model": "same"}
        training_contract = {"training": "same"}
        skip_data = {
            "active_splits": ["train", "val"],
            "test_evaluation_enabled": False,
            "manifests": {"train": "train.json", "val": "val.json"},
        }
        checkpoint = {
            "task": "ZInD_matching full end-to-end joint fine-tuning",
            "model_contract": model_contract,
            "training_contract": training_contract,
            "data_contract": dict(skip_data),
        }
        _validate_resume_contract(
            checkpoint, model_contract, training_contract, skip_data
        )

        changed = dict(skip_data)
        changed["active_splits"] = ["train", "val", "test"]
        changed["test_evaluation_enabled"] = True
        with self.assertRaisesRegex(ValueError, "data contract changed"):
            _validate_resume_contract(
                checkpoint, model_contract, training_contract, changed
            )

        full_data = {
            "active_splits": ["train", "val", "test"],
            "test_evaluation_enabled": True,
            "manifests": {
                "train": "train.json",
                "val": "val.json",
                "test": "test.json",
            },
        }
        legacy_full_data = {
            key: value
            for key, value in full_data.items()
            if key not in ("active_splits", "test_evaluation_enabled")
        }
        legacy_checkpoint = dict(checkpoint)
        legacy_checkpoint["data_contract"] = legacy_full_data
        _validate_resume_contract(
            legacy_checkpoint, model_contract, training_contract, full_data
        )


if __name__ == "__main__":
    unittest.main()
