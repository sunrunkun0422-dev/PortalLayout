import inspect
import math
import unittest
from pathlib import Path

import numpy as np

from tools.audit_zind_portal_geometry import (
    LineFit,
    _pair_predicted_inputs,
    _room_predicted_inputs,
    bi_to_zind,
    build_predicted_hypotheses,
    circular_indices,
    compare_context_lines,
    endpoint_gt_metrics,
    fit_line_tls,
    fit_pair_pose,
    line_endpoint_intersections,
    load_inputs,
    parse_args,
    quality_gate,
    ransac_line,
    reject_test_path,
)


def _line_points(angle_degrees, offset, count=30):
    angle = math.radians(angle_degrees)
    tangent = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    center = normal * offset
    coordinates = np.linspace(-1.5, 1.5, count)
    return center + coordinates[:, None] * tangent


class PortalLineGeometryTest(unittest.TestCase):
    def test_tls_recovers_arbitrary_rotated_wall(self):
        points = _line_points(37.0, 0.8)
        fit = fit_line_tls(points)
        recovered = math.atan2(fit.tangent[1], fit.tangent[0])
        error = abs((recovered - math.radians(37.0) + math.pi / 2) % math.pi - math.pi / 2)
        self.assertLess(error, 1e-10)
        self.assertLess(float(np.max(fit.residuals)), 1e-10)

    def test_ransac_handles_noise_and_outliers(self):
        rng = np.random.RandomState(9)
        clean = _line_points(-28.0, 0.55, count=45)
        noisy = clean + rng.normal(0.0, 0.006, clean.shape)
        outliers = np.asarray([[1.5, 1.5], [-1.3, 1.7], [0.1, -1.8]], dtype=np.float64)
        points = np.concatenate([noisy, outliers], axis=0)
        fit = ransac_line(points, threshold=0.025, min_inliers=20)
        recovered = math.atan2(fit.tangent[1], fit.tangent[0])
        error = abs((recovered - math.radians(-28.0) + math.pi / 2) % math.pi - math.pi / 2)
        self.assertLess(math.degrees(error), 0.5)
        self.assertGreaterEqual(int(np.count_nonzero(fit.inliers)), 43)
        self.assertLess(int(np.count_nonzero(fit.inliers)), len(points))

    def test_left_right_context_same_wall_agrees(self):
        left = fit_line_tls(_line_points(15.0, 0.7, count=12))
        right = fit_line_tls(_line_points(15.5, 0.705, count=12))
        report = compare_context_lines(left, right, room_radius=1.0)
        self.assertTrue(report["agrees"])
        self.assertLess(report["angle_difference_degrees"], 1.0)

    def test_left_right_context_corner_is_ambiguous(self):
        left = fit_line_tls(_line_points(0.0, 0.7, count=12))
        right = fit_line_tls(_line_points(78.0, 0.7, count=12))
        report = compare_context_lines(left, right, room_radius=1.0)
        self.assertFalse(report["agrees"])
        self.assertGreater(report["angle_difference_degrees"], 60.0)

    def test_wraparound_interval_is_inclusive_and_ordered(self):
        indices = circular_indices(252, 3, count=256)
        np.testing.assert_array_equal(indices, [252, 253, 254, 255, 0, 1, 2, 3])

    def test_ray_line_intersection_handles_wrap(self):
        points = np.asarray([[-1.0, -2.0], [1.0, -2.0]], dtype=np.float64)
        fit = fit_line_tls(points)
        endpoints, report = line_endpoint_intersections(fit, (252, 3))
        np.testing.assert_allclose(endpoints[:, 1], -2.0, atol=1e-10)
        self.assertTrue(np.all(np.asarray(report["ray_distances"]) > 0))

    def test_parallel_and_nonfinite_inputs_are_rejected(self):
        fit = LineFit(
            tangent=np.asarray([0.0, 1.0]),
            normal=np.asarray([1.0, 0.0]),
            offset=1.0,
            residuals=np.zeros(4),
            inliers=np.ones(4, dtype=bool),
            singular_values=np.asarray([1.0, 0.0]),
            source="test",
        )
        with self.assertRaisesRegex(ValueError, "parallel"):
            line_endpoint_intersections(fit, (0, 255), parallel_epsilon=0.02)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_line_tls([[0.0, 0.0], [float("nan"), 1.0]])

    def test_endpoint_metrics_accept_same_and_reversed_order(self):
        target = np.asarray([[-0.4, -0.7], [0.5, -0.6]], dtype=np.float64)
        direct = endpoint_gt_metrics(target, target)
        reversed_value = endpoint_gt_metrics(target[::-1], target)
        self.assertLess(direct["endpoint_rmse"], 1e-12)
        self.assertLess(reversed_value["endpoint_rmse"], 1e-12)
        self.assertEqual(reversed_value["endpoint_order"], "reversed")

    def test_bi_to_zind_is_x_reflection(self):
        points = np.asarray([[1.0, 2.0], [-3.0, -4.0]], dtype=np.float64)
        np.testing.assert_allclose(bi_to_zind(points), [[-1.0, 2.0], [3.0, -4.0]])
        np.testing.assert_allclose(bi_to_zind(bi_to_zind(points)), points)

    def test_pair_pose_is_b_to_a_and_not_inverted(self):
        endpoints_b = np.asarray([[-0.5, -1.0], [0.5, -1.0]], dtype=np.float64)
        yaw = math.radians(31.0)
        scale = 1.25
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        translation = np.asarray([0.7, -0.2], dtype=np.float64)
        endpoints_a = (scale * rotation @ endpoints_b.T).T + translation
        polygon_a = np.asarray([[-2, -2], [2, -2], [2, 2], [-2, 2]], dtype=np.float64)
        polygon_b = np.asarray([[-1, -2], [1, -2], [1, 0], [-1, 0]], dtype=np.float64)
        pose = fit_pair_pose(endpoints_a, endpoints_b, polygon_a, polygon_b)
        matrix = np.asarray(pose["T_B_to_A"], dtype=np.float64)
        homogeneous = np.concatenate([endpoints_b, np.ones((2, 1))], axis=1)
        transformed = (matrix @ homogeneous.T).T[:, :2]
        direct = np.max(np.abs(transformed - endpoints_a))
        reversed_error = np.max(np.abs(transformed - endpoints_a[::-1]))
        self.assertLess(min(direct, reversed_error), 1e-10)
        self.assertEqual(pose["transform_direction"], "B_to_A")


class PortalAuditSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = parse_args([])
        required = (args.rooms_json, args.pairs_json, args.group_manifest)
        if not all(Path(path).is_file() for path in required):
            raise unittest.SkipTest("requires the local frozen T3 room-group audit inputs")
        cls.inputs = load_inputs(args)
        cls.predicted = build_predicted_hypotheses(
            _room_predicted_inputs(cls.inputs["rooms"]), cls.inputs["pairs"]
        )

    def test_hypothesis_builder_and_quality_gate_have_no_gt_parameter(self):
        builder_names = set(inspect.signature(build_predicted_hypotheses).parameters)
        gate_names = set(inspect.signature(quality_gate).parameters)
        forbidden = {"gt", "ground_truth", "gt_pose", "gt_polygon", "gt_endpoints"}
        self.assertFalse(builder_names & forbidden)
        self.assertFalse(gate_names & forbidden)
        self.assertFalse(self.predicted["gt_available_to_hypothesis_generation"])

    def test_negative_pair_generates_no_portal_pose(self):
        self.assertEqual(len(self.predicted["negative_pairs"]), 1)
        value = next(iter(self.predicted["negative_pairs"].values()))
        self.assertEqual(value["best_candidate_pair"], [-1, -1])
        self.assertFalse(value["pose_generated"])
        self.assertEqual(value["methods_run"], [])

    def test_m0_reproduces_saved_frame_fixed_pose(self):
        for pair_id, pair in self.predicted["positive_pairs"].items():
            predicted_pose = np.asarray(pair["methods"]["M0"]["pose"]["T_B_to_A"])
            saved_pose = np.asarray(self.inputs["pairs"][pair_id]["relative_pose"]["T_B_to_A"])
            np.testing.assert_allclose(predicted_pose, saved_pose, atol=1e-10, rtol=0.0)

    def test_frozen_candidate_pairs_and_m1_are_deterministic(self):
        expected = {
            "0665_floor_03_pano_34_pano_37_3": [1, 0],
            "0665_floor_03_pano_34_pano_30_4": [0, 2],
        }
        self.assertEqual(
            {key: value["best_candidate_pair"] for key, value in self.predicted["positive_pairs"].items()},
            expected,
        )
        for room in self.predicted["room_state"].values():
            self.assertTrue(room["m1_report"]["validity"]["valid"])

    def test_test_paths_are_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "forbidden test data"):
            reject_test_path(Path("/tmp/data/test/rooms.json"), "rooms")


if __name__ == "__main__":
    unittest.main()
