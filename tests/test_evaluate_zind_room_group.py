import copy
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.evaluate_zind_room_group import (
    DISCLAIMER,
    _pair_inference_input,
    _pair_key,
    _reject_test_reference,
    _reject_test_checkpoint_metadata,
    _room_inference_inputs,
    _run_pair_prediction,
    attach_post_inference_evaluation,
    load_and_validate_inputs,
    metadata_report,
    parse_args,
    route_predicted_decision,
    strict_ground_truth_payload,
    validate_group_schema,
)


class ZInDRoomGroupMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = parse_args(["--metadata_only"])
        if not Path(cls.args.pair_manifest).is_file():
            raise unittest.SkipTest("requires the local ZInD_matching training manifest and room assets")
        cls.audit = load_and_validate_inputs(cls.args)

    def test_fixed_group_is_exact_three_room_chain(self):
        group = self.audit["group"]
        schema = validate_group_schema(group)

        self.assertEqual(group["split"], "train")
        self.assertEqual(group["purpose"], DISCLAIMER)
        self.assertEqual(group["house_id"], "0665")
        self.assertEqual(group["floor_id"], "floor_03")
        self.assertEqual(group["scale_meters_per_coordinate"], 3.618019586948089)
        self.assertEqual(
            group["room_ids"],
            ["complete_room_04", "complete_room_01", "complete_room_05"],
        )
        self.assertEqual(len(schema["all_pair_keys"]), 3)
        self.assertEqual(len(schema["positive_keys"]), 2)
        self.assertEqual(len(schema["negative_keys"]), 1)
        self.assertEqual(group["adjacency_matrix"], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def test_room_and_pair_selection_is_exact_and_deterministic(self):
        report_first = metadata_report(self.audit)
        report_second = metadata_report(self.audit)

        self.assertEqual(report_first, report_second)
        self.assertEqual(
            [value["pano_id"] for value in report_first["room_to_pano"].values()],
            ["pano_37", "pano_34", "pano_30"],
        )
        self.assertEqual(
            [pair["pair_id"] for pair in report_first["pairs"]],
            [
                "0665_floor_03_pano_34_pano_37_3",
                "0665_floor_03_pano_34_pano_30_4",
                "0665_floor_03_pano_37_pano_30_negative",
            ],
        )
        self.assertEqual([pair["is_match"] for pair in report_first["pairs"]], [True, True, False])
        self.assertTrue(all(pair["shared_interface_types"] == ["door", "door"] for pair in report_first["pairs"][:2]))
        self.assertFalse(report_first["model_loaded"])
        self.assertFalse(report_first["gpu_used"])
        self.assertFalse(report_first["test_data_accessed"])

    def test_inference_inputs_strip_all_ground_truth(self):
        room_inputs = _room_inference_inputs(self.audit)
        for value in room_inputs.values():
            self.assertNotIn("panorama", value)
            self.assertNotIn("layout_complete", value)
            self.assertNotIn("annotation", value)

        pair_id = self.audit["group"]["pair_ids"][_pair_key("complete_room_04", "complete_room_01")]
        pair_input = _pair_inference_input(self.audit["records"][pair_id])
        self.assertEqual(set(pair_input), {"pair_id", "room_a", "room_b", "pano_a", "pano_b"})
        self.assertNotIn("supervision", pair_input)
        parameter_names = set(inspect.signature(_run_pair_prediction).parameters)
        self.assertFalse(parameter_names & {"ground_truth", "gt", "record", "supervision"})

    def test_negative_gt_is_strictly_sanitized(self):
        pair_id = "0665_floor_03_pano_37_pano_30_negative"
        payload = strict_ground_truth_payload(
            self.audit["records"][pair_id],
            self.audit["annotations"],
            self.audit["schema"]["floor_scale"],
            self.args.endpoint_tolerance,
        )

        self.assertFalse(payload["is_match"])
        self.assertEqual(payload["shared_candidate_pairs"], [])
        self.assertIsNone(payload["relative_pose_gt"])
        self.assertIsNone(payload["joint_bev_target"])
        self.assertFalse(payload["pose_valid"])

    def test_test_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden test data"):
            _reject_test_reference(Path("/tmp/dataset/test/train.json"), "manifest")
        with self.assertRaisesRegex(ValueError, "metadata points to test"):
            _reject_test_checkpoint_metadata(
                {"manifests": {"train": "/data/train.json", "test": "/data/test.json"}}
            )


class PredictedOnlyDecisionTest(unittest.TestCase):
    def test_predicted_no_match_never_calls_fusion(self):
        calls = []

        def forbidden_fusion(index_a, index_b):
            calls.append((index_a, index_b))
            raise AssertionError("fusion must not be called")

        prediction = route_predicted_decision([-1, -1], forbidden_fusion)

        self.assertEqual(calls, [])
        self.assertFalse(prediction["is_match"])
        self.assertEqual(prediction["shared_candidate_pairs"], [])
        self.assertIsNone(prediction["relative_pose"])
        self.assertIsNone(prediction["joint_bev"])
        self.assertFalse(prediction["pose_valid"])
        self.assertFalse(prediction["joint_bev_generated"])

    def test_predicted_match_calls_fusion_once(self):
        calls = []

        def fusion(index_a, index_b):
            calls.append((index_a, index_b))
            return {
                "shared_candidate_pairs": [[index_a, index_b]],
                "relative_pose": {"T_B_to_A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                "pose_valid": True,
                "joint_bev": {"rooms": ["A", "B"]},
                "joint_bev_generated": True,
            }

        prediction = route_predicted_decision([2, 4], fusion)

        self.assertEqual(calls, [(2, 4)])
        self.assertTrue(prediction["is_match"])
        self.assertTrue(prediction["pose_valid"])
        self.assertTrue(prediction["joint_bev_generated"])
        self.assertEqual(prediction["shared_candidate_pairs"], [[2, 4]])

    def test_changing_gt_cannot_change_frozen_prediction(self):
        prediction = route_predicted_decision([-1, -1], lambda *_: {})
        frozen = copy.deepcopy(prediction)
        first = attach_post_inference_evaluation(
            prediction, {"is_match": True, "target": [0, 0]}
        )
        second = attach_post_inference_evaluation(
            prediction, {"is_match": False, "target": None}
        )

        self.assertEqual(prediction, frozen)
        self.assertEqual(first["prediction"], frozen)
        self.assertEqual(second["prediction"], frozen)
        self.assertNotEqual(first["ground_truth"], second["ground_truth"])
        self.assertEqual(first["ground_truth_usage"], "post-inference evaluation only")

    def test_m2_hard_failure_preserves_match_without_pose_or_joint(self):
        prediction = route_predicted_decision(
            [1, 0],
            lambda index_a, index_b: {
                "shared_candidate_pairs": [[index_a, index_b]],
                "relative_pose": None,
                "predicted_relative_pose": None,
                "pose_valid": False,
                "joint_bev": None,
                "joint_bev_generated": False,
                "portal_geometry_source_A": "candidate_local_m2",
                "portal_geometry_source_B": "candidate_local_m2",
                "portal_geometry_valid_A": False,
                "portal_geometry_valid_B": True,
                "portal_geometry_failure_reason_A": "line fit degenerate",
                "failure_stage": "candidate_local_portal_geometry",
                "failure_reason": "M2 hard geometry failure on side A",
            },
        )
        self.assertTrue(prediction["is_match"])
        self.assertFalse(prediction["pose_valid"])
        self.assertIsNone(prediction["relative_pose"])
        self.assertIsNone(prediction["joint_bev"])
        self.assertEqual(prediction["shared_candidate_pairs"], [[1, 0]])
        self.assertEqual(
            prediction["failure_stage"], "candidate_local_portal_geometry"
        )


class FrozenMatcherRoutingTest(unittest.TestCase):
    class Candidate:
        token_start = 12
        token_end = 18

        def to_json(self):
            return {"tokenStart": 12, "tokenEnd": 18, "candidateIndex": 0}

    @staticmethod
    def _state():
        return {
            "_feature": torch.zeros((1, 2, 3)),
            "_enclosed": torch.zeros((1, 256)),
            "_extended": torch.zeros((1, 256)),
            "_candidate_masks": torch.ones((1, 1, 256), dtype=torch.bool),
            "_candidate_valid": torch.ones((1, 1), dtype=torch.bool),
            "_opening_candidates": [FrozenMatcherRoutingTest.Candidate()],
        }

    @staticmethod
    def _matcher(best):
        class Matcher:
            def __call__(self, *args, **kwargs):
                return {
                    "best_candidate_pair": torch.tensor([best], dtype=torch.long),
                    "candidate_pair_score": torch.tensor([[[0.73125]]]),
                    "pair_match_probability": torch.tensor([0.8125]),
                    "pair_no_match_probability": torch.tensor([0.1875]),
                }

        return Matcher()

    def test_m2_does_not_change_candidate_pair_or_matcher_score(self):
        args = SimpleNamespace(portal_geometry_mode="candidate_local_m2", top_k=1)
        pair_input = {
            "pair_id": "pair",
            "room_a": "A",
            "room_b": "B",
            "pano_a": "pano_a",
            "pano_b": "pano_b",
        }
        payload = {
            "shared_candidate_pairs": [[0, 0]],
            "relative_pose": {"T_B_to_A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "pose_valid": True,
            "joint_bev": {"rooms": []},
            "joint_bev_generated": True,
            "portal_geometry_source_A": "candidate_local_m2",
            "portal_geometry_source_B": "candidate_local_m2",
            "portal_geometry_valid_A": True,
            "portal_geometry_valid_B": True,
        }
        with patch(
            "tools.evaluate_zind_room_group._candidate_local_pair_fusion",
            return_value=(payload, {"native_joint": None, "raw_joint": None}),
        ) as fusion:
            prediction, artifacts = _run_pair_prediction(
                args,
                self._matcher([0, 0]),
                pair_input,
                {"A": self._state(), "B": self._state()},
            )
        fusion.assert_called_once()
        self.assertEqual(prediction["best_candidate_pair"], [0, 0])
        self.assertEqual(prediction["shared_candidate_pairs"], [[0, 0]])
        self.assertAlmostEqual(prediction["candidate_pair_scores"][0][0], 0.73125)
        self.assertAlmostEqual(
            float(artifacts["matcher_outputs"]["candidate_pair_score"][0, 0, 0]),
            0.73125,
        )

    def test_negative_matcher_decision_never_calls_m2(self):
        args = SimpleNamespace(portal_geometry_mode="candidate_local_m2", top_k=1)
        pair_input = {
            "pair_id": "negative",
            "room_a": "A",
            "room_b": "B",
            "pano_a": "pano_a",
            "pano_b": "pano_b",
        }
        with patch(
            "tools.evaluate_zind_room_group._candidate_local_pair_fusion"
        ) as fusion:
            prediction, _ = _run_pair_prediction(
                args,
                self._matcher([-1, -1]),
                pair_input,
                {"A": self._state(), "B": self._state()},
            )
        fusion.assert_not_called()
        self.assertFalse(prediction["is_match"])
        self.assertEqual(prediction["best_candidate_pair"], [-1, -1])
        self.assertIsNone(prediction["relative_pose"])
        self.assertIsNone(prediction["joint_bev"])

    def test_polygon_mapper_remains_the_default(self):
        self.assertEqual(parse_args([]).portal_geometry_mode, "polygon_mapper")


if __name__ == "__main__":
    unittest.main()
