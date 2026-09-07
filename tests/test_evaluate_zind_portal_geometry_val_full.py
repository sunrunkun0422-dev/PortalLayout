import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools import evaluate_zind_portal_geometry_val_full as full
from tools.evaluate_zind_portal_geometry_val import DEFAULT_VAL_MANIFEST, load_json


class FullManifestContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_VAL_MANIFEST.is_file():
            raise unittest.SkipTest("requires the local ZInD_matching validation manifest")
        cls.val = load_json(DEFAULT_VAL_MANIFEST)
        cls.manifest = full.build_full_manifest(cls.val, DEFAULT_VAL_MANIFEST)

    def test_contains_every_pair_once_in_stable_order(self):
        expected = sorted(str(value["id"]) for value in self.val["pairs"])
        self.assertEqual(self.manifest["pair_ids"], expected)
        self.assertEqual(len(expected), len(set(expected)))
        self.assertEqual(self.manifest["counts"]["pair_count"], len(expected))
        self.assertEqual(
            self.manifest["counts"]["positive"]
            + self.manifest["counts"]["negative"],
            len(expected),
        )
        self.assertFalse(self.manifest["test_data_accessed"])

    def test_validation_rejects_reordering_or_filtering(self):
        changed = copy.deepcopy(self.manifest)
        changed["pair_ids"] = list(reversed(changed["pair_ids"]))
        with self.assertRaisesRegex(ValueError, "differs"):
            full.validate_full_manifest(changed, self.manifest)

    def test_threshold_has_no_command_line_override(self):
        args = full.parse_args([])
        self.assertIsNone(args.opening_probability_threshold)
        with self.assertRaises(SystemExit):
            full.parse_args(["--opening_probability_threshold", "0.5"])


class MatcherParityDefinitionTest(unittest.TestCase):
    @staticmethod
    def _record(is_match, predicted_match, recalled=False, correct=False):
        return {
            "ground_truth": {"is_match": is_match},
            "prediction": {
                "is_match": predicted_match,
                "pair_match_probability": 0.8 if predicted_match else 0.2,
                "pair_no_match_probability": 0.2 if predicted_match else 0.8,
                "matcher_score_max_abs_diff": 0.0,
                "matcher_probability_max_abs_diff": 0.0,
                "best_candidate_identical": True,
            },
            "evaluation": {
                "gt_candidate_recalled": recalled,
                "selected_candidate_correct": correct,
            },
        }

    def test_conditional_top1_uses_recalled_positive_denominator(self):
        records = [
            self._record(True, True, True, True),
            self._record(True, True, True, False),
            self._record(True, False, False, False),
            self._record(False, False),
            self._record(False, True),
        ]
        value = full.matching_summary(records)
        self.assertEqual(value["counts"], {"TP": 2, "TN": 1, "FP": 1, "FN": 1})
        self.assertAlmostEqual(value["positive_candidate_recall"], 2.0 / 3.0)
        self.assertAlmostEqual(value["conditional_candidate_top1"], 0.5)
        self.assertAlmostEqual(value["end_to_end_candidate_top1"], 1.0 / 3.0)
        self.assertEqual(
            value["matcher_probability_max_abs_diff_across_geometry_modes"], 0.0
        )


class ScaleDiagnosticGeometryTest(unittest.TestCase):
    def test_fixed_scale_transform_respects_requested_scale_and_portal_alignment(self):
        endpoints_a = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
        endpoints_b = np.asarray([[-0.5, 0.0], [0.5, 0.0]])
        polygon_a = np.asarray([[-2.0, 0.2], [2.0, 0.2], [2.0, 2.0], [-2.0, 2.0]])
        polygon_b = np.asarray([[-1.0, -0.2], [1.0, -0.2], [1.0, -1.0], [-1.0, -1.0]])
        matrix, mapping = full._fixed_scale_matrix(
            endpoints_a, endpoints_b, polygon_a, polygon_b, 2.0
        )
        self.assertIn(mapping, {"same", "reversed"})
        self.assertAlmostEqual(np.linalg.det(matrix[:2, :2]), 4.0)
        transformed = full._transform(endpoints_b, matrix)
        target = endpoints_a if mapping == "same" else endpoints_a[::-1]
        np.testing.assert_allclose(transformed, target, atol=1e-12)

    def test_scale_diagnostic_is_not_defined_for_negative_or_invalid_m2(self):
        negative = {"ground_truth": {"is_match": False}}
        self.assertIsNone(full.scale_diagnostic(negative))
        invalid = {
            "ground_truth": {"is_match": True},
            "prediction": {
                "geometry_modes": {
                    "candidate_local_m2": {
                        "pose_valid": False,
                        "portal_geometry_A": None,
                        "portal_geometry_B": None,
                    }
                }
            },
            "scale_diagnostic_inputs": None,
        }
        self.assertIsNone(full.scale_diagnostic(invalid))


class ResumeAndBootstrapContractTest(unittest.TestCase):
    @staticmethod
    def _context():
        return {
            "checkpoint_sha256": "checkpoint",
            "full_sha256": "full",
            "full": {"source_manifest": {"sha256": "val"}},
            "m2": {"sha256": "m2"},
        }

    @staticmethod
    def _metadata():
        return {
            "checkpoint": {"sha256": "checkpoint"},
            "full_manifest": {"sha256": "full"},
            "val_manifest": {"sha256": "val"},
            "m2": {"sha256": "m2"},
            "opening_threshold": {"value": full.EXPECTED_OPENING_THRESHOLD},
        }

    def test_resume_requires_all_frozen_identities(self):
        full._validate_resume_metadata(self._metadata(), self._context())
        changed = self._metadata()
        changed["m2"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "M2 config"):
            full._validate_resume_metadata(changed, self._context())

    def test_house_bootstrap_is_repeatable_and_includes_scale_conditioned_success(self):
        rows = []
        for index, house in enumerate(("h1", "h2")):
            row = {
                "pair_id": f"p{index}",
                "house_id": house,
                "delta_m2_minus_g1_yaw": float(index - 1),
                "delta_m2_minus_g1_translation_meters": float(index) / 10.0,
                "delta_m2_minus_g1_log_scale": float(index) / 20.0,
                "delta_m2_minus_g1_bev_frame": float(index) / 30.0,
                "delta_m2_minus_g1_bev_raw": float(index) / 40.0,
            }
            for yaw, translation in full.POSE_THRESHOLDS:
                suffix = f"{yaw:g}_{translation:g}"
                row[f"m2_success_{suffix}"] = index == 0
                row[f"g1_success_{suffix}"] = index == 1
                for scale in full.SCALE_THRESHOLDS:
                    scale_suffix = f"{suffix}_scale_{100 * scale:g}pct"
                    row[f"m2_success_{scale_suffix}"] = index == 0
                    row[f"g1_success_{scale_suffix}"] = index == 1
            rows.append(row)
        with mock.patch.object(full, "BOOTSTRAP_RESAMPLES", 25):
            first = full.clustered_bootstrap(rows)
            second = full.clustered_bootstrap(rows)
        self.assertEqual(first, second)
        self.assertIn(
            "success_rate_delta_10_0.5_scale_10pct_m2_minus_g1",
            first["metrics"],
        )


if __name__ == "__main__":
    unittest.main()
