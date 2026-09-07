import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_zind_portal_geometry_val import (
    AnnotationResolver,
    DEFAULT_VAL_MANIFEST,
    GEOMETRY_MODES,
    PURPOSE,
    _frozen_file,
    _house_first_select,
    _m2_contract,
    _pair_file,
    atomic_write_json,
    build_sample_manifest,
    build_summary,
    geometry_fanout,
    infer_frozen_prediction,
    load_completed_records,
    load_json,
    parse_args,
    _report_markdown,
    reject_test_path,
    rewrite_per_pair_jsonl,
    validate_args,
    validate_sample_manifest,
)
from utils.candidate_local_portal_geometry import CandidateLocalPortalConfig


class DeterministicValSmokeSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_VAL_MANIFEST.is_file():
            raise unittest.SkipTest("requires the local ZInD_matching validation manifest")
        cls.val = load_json(DEFAULT_VAL_MANIFEST)
        cls.sample = build_sample_manifest(cls.val, DEFAULT_VAL_MANIFEST)

    def test_contract_is_exact_32_plus_32_with_broad_house_coverage(self):
        self.assertEqual(self.sample["split"], "val")
        self.assertEqual(self.sample["purpose"], PURPOSE)
        self.assertEqual(self.sample["counts"]["pair_count"], 64)
        self.assertEqual(self.sample["counts"]["positive"], 32)
        self.assertEqual(self.sample["counts"]["negative"], 32)
        self.assertEqual(self.sample["counts"]["house_count"], 63)
        self.assertEqual(len(self.sample["pair_ids"]), 64)
        self.assertEqual(len(set(self.sample["pair_ids"])), 64)
        self.assertEqual(
            self.sample["positive_interface_types"],
            {"door-door": 31, "opening-door": 1},
        )

    def test_selection_is_repeatable_and_manifest_contains_only_references(self):
        repeated = build_sample_manifest(self.val, DEFAULT_VAL_MANIFEST)
        self.assertEqual(self.sample, repeated)
        self.assertNotIn("pairs", self.sample)
        self.assertFalse(self.sample["selection"]["model_outputs_used"])
        self.assertFalse(self.sample["selection"]["geometry_or_quality_used"])
        validate_sample_manifest(self.sample, repeated)

    def test_posthoc_annotations_allow_explicitly_missing_metric_scale(self):
        records = {value["id"]: value for value in self.val["pairs"]}
        resolver = AnnotationResolver(Path(self.val["dataRoot"]))
        ground_truth = [
            resolver.pair_ground_truth(records[pair_id])
            for pair_id in self.sample["pair_ids"]
        ]
        self.assertEqual(
            sum(not value["metric_translation_available"] for value in ground_truth),
            8,
        )
        self.assertEqual(
            sum(
                value["is_match"] and not value["metric_translation_available"]
                for value in ground_truth
            ),
            2,
        )
        self.assertTrue(
            all(
                value["relative_pose"] is not None
                and value["joint_bev_target"] is not None
                for value in ground_truth
                if value["is_match"]
            )
        )

    def test_house_first_selection_uses_stable_id_not_input_order(self):
        records = [
            {"id": "z", "scene_id": "h1"},
            {"id": "a", "scene_id": "h1"},
            {"id": "b", "scene_id": "h2"},
        ]
        first = _house_first_select(records, "positive", 3, 123)
        second = _house_first_select(list(reversed(records)), "positive", 3, 123)
        self.assertEqual([value["id"] for value in first], [value["id"] for value in second])
        self.assertEqual({value["id"] for value in first[:2]}, {"a", "b"})


class ValOnlyAndGroundTruthIsolationTest(unittest.TestCase):
    def test_command_line_does_not_offer_test_split(self):
        with self.assertRaises(SystemExit):
            parse_args(["--split", "test"])

    def test_test_paths_and_contract_mutations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden test data"):
            reject_test_path(Path("/tmp/ZInD_matching/test.json"), "manifest")
        args = parse_args(["--seed", "124"])
        with self.assertRaisesRegex(ValueError, "fixed to seed=123"):
            validate_args(args)
        args = parse_args(["--geometry_modes", "candidate_local_m2"])
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            validate_args(args)

    def test_prediction_and_geometry_signatures_accept_no_ground_truth(self):
        forbidden = {"gt", "ground_truth", "supervision", "record", "annotation"}
        self.assertFalse(
            forbidden & set(inspect.signature(infer_frozen_prediction).parameters)
        )
        self.assertFalse(forbidden & set(inspect.signature(geometry_fanout).parameters))

    def test_no_match_fans_out_without_pose(self):
        frozen = {
            "is_match": False,
            "best_candidate_pair": [-1, -1],
            "candidate_pair_scores": [[0.25]],
            "pair_match_probability": 0.1,
            "pair_no_match_probability": 0.9,
        }
        output = geometry_fanout(frozen)
        self.assertEqual(tuple(output["modes"]), GEOMETRY_MODES)
        self.assertEqual(output["matcher_score_max_abs_diff"], 0.0)
        self.assertTrue(output["best_candidate_identical"])
        self.assertFalse(output["gt_available_to_geometry"])
        for geometry in output["modes"].values():
            self.assertFalse(geometry["attempted"])
            self.assertFalse(geometry["pose_valid"])
            self.assertIsNone(geometry["relative_pose"])

    def test_m2_contract_is_the_unchanged_production_default(self):
        first = _m2_contract()
        second = _m2_contract()
        self.assertEqual(first, second)
        self.assertEqual(first["config"], CandidateLocalPortalConfig().to_json())
        self.assertFalse(first["tuned_on_val"])


class InterruptionSafetyTest(unittest.TestCase):
    def test_atomic_pair_artifacts_drive_ordered_resume_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["pair_a", "pair_b"]
            records = {
                "pair_b": {"pair_id": "pair_b", "split": "val", "value": 2},
                "pair_a": {"pair_id": "pair_a", "split": "val", "value": 1},
            }
            atomic_write_json(_pair_file(root, 1, "pair_b"), records["pair_b"])
            atomic_write_json(_pair_file(root, 0, "pair_a"), records["pair_a"])
            atomic_write_json(
                _frozen_file(root, 0, "pair_a"),
                {"pair_id": "pair_a", "model_forward_count": 1},
            )
            completed = load_completed_records(root, ids)
            self.assertEqual(set(completed), set(ids))
            rewrite_per_pair_jsonl(root, ids, completed)
            lines = [json.loads(value) for value in (root / "per_pair.jsonl").read_text().splitlines()]
            self.assertEqual([value["pair_id"] for value in lines], ids)
            self.assertEqual(
                load_json(_frozen_file(root, 0, "pair_a"))["model_forward_count"], 1
            )


class SummaryContractTest(unittest.TestCase):
    @staticmethod
    def _metrics(valid):
        value = 1.0 if valid else None
        bev = None
        if valid:
            bev = {
                "instance_A_overlap": 0.7,
                "instance_B_overlap": 0.6,
                "instance_mean_overlap": 0.65,
                "polygon_A_valid": True,
                "polygon_B_valid": True,
            }
        return {
            "yaw_error_degrees": value,
            "translation_error": value,
            "translation_error_meters": value,
            "absolute_log_scale_error": 0.02 if valid else None,
            "relative_scale_error": 0.02 if valid else None,
            "predicted_scale": 1.02 if valid else None,
            "gt_scale": 1.0 if valid else None,
            "bev": {"frame_preserving": bev, "pre_manhattan": bev},
        }

    @classmethod
    def _geometry(cls, mode, valid, with_portals=False):
        portal = None
        if with_portals:
            portal = {
                "primary_variant": "ransac",
                "token_interval": [10, 20],
                "predicted_only_quality": {
                    "support_point_count": 8,
                    "inlier_ratio": 0.9,
                    "normalized_residual": 0.01,
                    "angular_dispersion_degrees": 5.0,
                    "ray_intersection_condition": 0.3,
                    "opening_width": 1.0,
                    "reliability_score": 0.8,
                    "width_plausible": True,
                    "quality_gate_pass": True,
                    "corner_ambiguity": False,
                },
            }
        return {
            "mode": mode,
            "attempted": valid,
            "pose_valid": valid,
            "hard_failure": False,
            "failure_reason": None,
            "portal_geometry_A": portal,
            "portal_geometry_B": portal,
            "metrics": cls._metrics(valid),
        }

    def test_all_strata_and_report_are_defined_without_survivor_only_denominators(self):
        positive = {
            "pair_id": "p",
            "house_id": "h1",
            "ground_truth": {
                "is_match": True,
                "gt_scale_is_approximately_unit": True,
            },
            "prediction": {
                "is_match": True,
                "matcher_score_max_abs_diff": 0.0,
                "best_candidate_identical": True,
                "geometry_modes": {
                    mode: self._geometry(
                        mode, True, with_portals=mode == "candidate_local_m2"
                    )
                    for mode in GEOMETRY_MODES
                },
            },
            "evaluation": {
                "gt_candidate_recalled": True,
                "selected_candidate_correct": True,
            },
        }
        negative = {
            "pair_id": "n",
            "house_id": "h2",
            "ground_truth": {
                "is_match": False,
                "gt_scale_is_approximately_unit": None,
            },
            "prediction": {
                "is_match": False,
                "matcher_score_max_abs_diff": 0.0,
                "best_candidate_identical": True,
                "geometry_modes": {
                    mode: self._geometry(mode, False) for mode in GEOMETRY_MODES
                },
            },
            "evaluation": {
                "gt_candidate_recalled": False,
                "selected_candidate_correct": False,
            },
        }
        summary = build_summary([positive, negative], {"smoke": True})
        self.assertEqual(summary["set_counts"]["S0_all_pairs"], 2)
        self.assertEqual(summary["set_counts"]["S1_all_gt_positive"], 1)
        self.assertEqual(summary["set_counts"]["S3_gt_positive_selected_candidate_correct"], 1)
        self.assertEqual(summary["matching"]["counts"], {"TP": 1, "TN": 1, "FP": 0, "FN": 0})
        self.assertEqual(
            summary["geometry"]["candidate_local_m2"]["S1_all_gt_positive"]
            ["yaw_error_degrees"]["coverage_aware_threshold_rates"]["lt_5"]
            ["denominator"],
            1,
        )
        self.assertIn("No confidence interval", _report_markdown(summary))


if __name__ == "__main__":
    unittest.main()
