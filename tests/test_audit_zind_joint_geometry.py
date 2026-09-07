import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit_zind_joint_geometry import (
    apply_transform,
    audit_negative_prediction,
    circular_yaw_error,
    compose_canonical_to_local,
    finite_tree,
    fit_similarity_from_endpoints,
    formal_prediction_payload,
    homogeneous_rotation,
    reject_test_data_path,
    tokens_in_interval,
    transform_scale,
    transform_yaw,
)


class SimilarityGeometryTest(unittest.TestCase):
    def test_known_similarity_recovers_b_to_a(self):
        endpoints_b = np.asarray([[0.0, 0.0], [2.0, 0.0]])
        expected = homogeneous_rotation(math.radians(31.0))
        expected[:2, :2] *= 1.7
        expected[:2, 2] = [3.0, -2.0]
        endpoints_a = apply_transform(endpoints_b, expected)

        recovered = fit_similarity_from_endpoints(endpoints_b, endpoints_a, 1.7)

        np.testing.assert_allclose(recovered, expected, atol=1e-12)
        np.testing.assert_allclose(apply_transform(endpoints_b, recovered), endpoints_a)
        self.assertAlmostEqual(transform_scale(recovered), 1.7)

    def test_same_and_reversed_endpoint_direction(self):
        endpoints_b = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        endpoints_a = np.asarray([[2.0, 1.0], [2.0, 2.0]])
        same = fit_similarity_from_endpoints(endpoints_b, endpoints_a, 1.0)
        reversed_mapping = fit_similarity_from_endpoints(endpoints_b, endpoints_a[::-1], 1.0)

        self.assertAlmostEqual(math.degrees(transform_yaw(same)), 90.0)
        self.assertAlmostEqual(math.degrees(transform_yaw(reversed_mapping)), -90.0)
        np.testing.assert_allclose(apply_transform(endpoints_b, same), endpoints_a)
        np.testing.assert_allclose(
            apply_transform(endpoints_b, reversed_mapping), endpoints_a[::-1]
        )

    def test_unit_scale_is_not_width_ratio(self):
        endpoints_b = np.asarray([[0.0, 0.0], [2.0, 0.0]])
        endpoints_a = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        unit = fit_similarity_from_endpoints(endpoints_b, endpoints_a, 1.0)
        width = fit_similarity_from_endpoints(endpoints_b, endpoints_a, 0.5)

        self.assertEqual(transform_scale(unit), 1.0)
        self.assertEqual(transform_scale(width), 0.5)
        self.assertGreater(
            np.linalg.norm(apply_transform(endpoints_b, unit) - endpoints_a), 0.0
        )
        np.testing.assert_allclose(apply_transform(endpoints_b, width), endpoints_a)

    def test_b_to_a_direction_is_not_inverted(self):
        endpoints_b = np.asarray([[1.0, 0.0], [2.0, 0.0]])
        endpoints_a = np.asarray([[5.0, 2.0], [6.0, 2.0]])
        b_to_a = fit_similarity_from_endpoints(endpoints_b, endpoints_a, 1.0)

        np.testing.assert_allclose(apply_transform(endpoints_b, b_to_a), endpoints_a)
        self.assertFalse(
            np.allclose(apply_transform(endpoints_a, b_to_a), endpoints_b)
        )

    def test_circular_yaw_error_wraps(self):
        error = circular_yaw_error(math.radians(179.0), math.radians(-179.0))
        self.assertAlmostEqual(math.degrees(error), 2.0)


class CoordinateFrameTest(unittest.TestCase):
    def test_canonical_local_composition_and_inverse_roundtrip(self):
        transform_canonical = homogeneous_rotation(math.radians(20.0))
        transform_canonical[:2, 2] = [0.7, -1.1]
        canonical_to_local_a = homogeneous_rotation(math.radians(35.0))
        canonical_to_local_b = homogeneous_rotation(math.radians(-12.0))
        transform_local = compose_canonical_to_local(
            transform_canonical, canonical_to_local_a, canonical_to_local_b
        )
        point_b_local = np.asarray([[2.0, -0.5]])

        point_b_canonical = apply_transform(
            point_b_local, np.linalg.inv(canonical_to_local_b)
        )
        point_a_canonical = apply_transform(point_b_canonical, transform_canonical)
        expected_a_local = apply_transform(point_a_canonical, canonical_to_local_a)

        np.testing.assert_allclose(
            apply_transform(point_b_local, transform_local), expected_a_local, atol=1e-12
        )
        np.testing.assert_allclose(
            apply_transform(
                apply_transform(point_b_local, canonical_to_local_b),
                np.linalg.inv(canonical_to_local_b),
            ),
            point_b_local,
            atol=1e-12,
        )

    def test_wraparound_token_interval_is_inclusive(self):
        tokens = tokens_in_interval(250, 3, 256)
        np.testing.assert_array_equal(
            tokens, np.asarray([250, 251, 252, 253, 254, 255, 0, 1, 2, 3])
        )


class OracleSafetyTest(unittest.TestCase):
    def test_negative_pair_does_not_generate_pose(self):
        result = audit_negative_prediction(
            {
                "pair_id": "negative",
                "is_match": False,
                "best_candidate_pair": [-1, -1],
                "relative_pose": None,
                "joint_bev": None,
            }
        )
        self.assertFalse(result["pose_generated"])
        self.assertFalse(result["oracle_pose_generated"])

    def test_formal_payload_contains_no_ground_truth(self):
        rooms = {
            "A": {
                "single_room_layout": {"predicted_polygon_zind_local_xy": [[0, 0]]},
                "opening_prediction": {
                    "intervals": [[250, 3]],
                    "probability": [0.0] * 256,
                },
                "ground_truth": {"forbidden": True},
            },
            "B": {
                "single_room_layout": {"predicted_polygon_zind_local_xy": [[1, 1]]},
                "opening_prediction": {
                    "intervals": [[10, 20]],
                    "probability": [0.0] * 256,
                },
                "ground_truth": {"forbidden": True},
            },
        }
        payload = formal_prediction_payload(
            {
                "pair_id": "p",
                "room_a": "A",
                "room_b": "B",
                "best_candidate_pair": [0, 0],
                "ground_truth": {"forbidden": True},
            },
            rooms,
        )

        self.assertNotIn("ground_truth", payload)
        self.assertNotIn("oracle", payload)
        self.assertNotIn("GT", repr(payload))

    def test_nonfinite_output_is_rejected(self):
        self.assertTrue(finite_tree({"value": [0.0, 1.0]}))
        self.assertFalse(finite_tree({"value": float("nan")}))
        self.assertFalse(finite_tree({"value": float("inf")}))

    def test_test_manifest_path_is_rejected_without_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "test.json"
            # The file deliberately does not exist: rejection happens from the path alone.
            with self.assertRaisesRegex(ValueError, "forbidden test data"):
                reject_test_data_path(forbidden, "manifest")


if __name__ == "__main__":
    unittest.main()
