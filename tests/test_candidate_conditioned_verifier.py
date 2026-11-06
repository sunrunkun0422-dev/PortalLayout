import unittest
from pathlib import Path

import numpy as np

from tools.preflight_zind_bev_verifier import (
    SafeFeatures,
    feature_schema,
    group_feature_rows,
    house_split,
    reject_test_path,
)
from portallayout.utils.candidate_conditioned_verifier import (
    endpoint_consistency,
    manifest_target_pair,
    pad_candidate_rows,
    predicted_polygon_features,
    stable_topn,
    transform_points,
)
from portallayout.geometry.candidate_local_portal_geometry import (
    estimate_candidate_local_portal_geometry,
)


class CandidateConditionedVerifierTest(unittest.TestCase):
    def test_topn_stable_sort_and_padding_mask(self):
        ranked = stable_topn([[0.5, 0.5], [0.7, 0.5]], 3)
        self.assertEqual(
            [(row.candidate_a, row.candidate_b) for row in ranked],
            [(1, 0), (0, 0), (0, 1)],
        )
        padded, mask = pad_candidate_rows([[1.0, 2.0], [3.0, 4.0]], 3, 2)
        np.testing.assert_array_equal(mask, [True, True, False])
        np.testing.assert_allclose(padded[2], [0.0, 0.0])

    def test_b_to_a_direction_and_endpoint_mappings(self):
        matrix = np.asarray([[0.0, -2.0, 3.0], [2.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(transform_points([[1.0, 0.0]], matrix), [[3.0, 1.0]])
        endpoints_a = np.asarray([[0.0, 0.0], [2.0, 0.0]])
        same = endpoint_consistency(endpoints_a, endpoints_a, np.eye(3), "same")
        reversed_value = endpoint_consistency(endpoints_a, endpoints_a[::-1], np.eye(3), "reversed")
        self.assertAlmostEqual(same["endpoint_alignment_residual"], 0.0)
        self.assertAlmostEqual(reversed_value["endpoint_alignment_residual"], 0.0)

    def test_m2_hard_failure_and_finite_missing_mask(self):
        result = estimate_candidate_local_portal_geometry(
            np.zeros((256, 2)), (10, 20), 256, np.ones(256)
        )
        self.assertFalse(result.geometry_valid)
        self.assertIsNotNone(result.failure_reason)
        features = SafeFeatures()
        features.set("t3_raw_candidate_score", float("nan"))
        features.set("t3_candidate_rank", 2)
        vector = features.vector()
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(features.missing["t3_raw_candidate_score"], 1.0)
        self.assertEqual(features.missing["t3_candidate_rank"], 0.0)

    def test_polygon_intersection_and_ab_exchange(self):
        polygon_a = [[0, 0], [2, 0], [2, 2], [0, 2]]
        polygon_b = [[1, 1], [3, 1], [3, 3], [1, 3]]
        portal = [[1, 0], [2, 0]]
        first, failure_a = predicted_polygon_features(polygon_a, polygon_b, portal)
        second, failure_b = predicted_polygon_features(polygon_b, polygon_a, portal[::-1])
        self.assertIsNone(failure_a)
        self.assertIsNone(failure_b)
        self.assertAlmostEqual(first["intersection_area"], 1.0)
        self.assertAlmostEqual(first["union_area"], 7.0)
        self.assertAlmostEqual(first["polygon_iou"], 1.0 / 7.0)
        self.assertAlmostEqual(first["intersection_area"], second["intersection_area"])
        self.assertAlmostEqual(first["centroid_distance"], second["centroid_distance"])

    def test_manifest_candidate_label_and_absent_target(self):
        target, audit = manifest_target_pair(
            [10, 20], [30, 40], [[0, 5], [10, 20]], [[30, 40], [80, 90]]
        )
        self.assertEqual(target, (1, 0))
        self.assertTrue(audit["mapped"])
        absent, audit = manifest_target_pair(
            [10, 20], [30, 40], [[100, 110]], [[120, 130]]
        )
        self.assertIsNone(absent)
        self.assertFalse(audit["mapped"])

    def test_positive_target_absent_and_negative_no_valid_candidate(self):
        base = {
            "pair_id": "pair", "house_id": "house", "candidate_rank": 1,
            "candidate_is_true": False,
        }
        grouped = group_feature_rows([base])
        self.assertFalse(grouped[0]["shortlist_has_target"])
        self.assertIsNone(grouped[0]["target_index"])
        # Phase-B semantics: a negative has no true candidate by construction.
        negative_rows = [{**base, "pair_id": "negative", "pair_is_adjacent": False}]
        negative = group_feature_rows(negative_rows)[0]
        self.assertIsNone(negative["target_index"])

    def test_house_isolation_no_gt_no_test_and_seed_reproducibility(self):
        groups = [
            {"pair_id": f"p{i}", "house_id": f"h{i}", "rows": [], "target_index": 0,
             "shortlist_has_target": True}
            for i in range(12)
        ]
        first = house_split(groups)
        second = house_split(groups)
        self.assertEqual(first, second)
        self.assertFalse(set(first["subtrain_houses"]) & set(first["train_dev_houses"]))
        schema = feature_schema()
        self.assertTrue(schema["predicted_only"])
        self.assertEqual(schema["gt_feature_fields"], [])
        with self.assertRaises(ValueError):
            reject_test_path(Path("/tmp/ZInD_matching/test.json"), "manifest")


if __name__ == "__main__":
    unittest.main()
