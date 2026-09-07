import inspect
import math
import unittest

import numpy as np
import torch

from postprocessing.post_process import (
    estimate_manhattan_heading,
    frame_preserving_manhattan,
    post_process,
    rotate_floor_xz,
)
from tools.debug_cross_scene_flow import prediction_to_layout
from tools.evaluate_zind_room_group import parse_args
from utils.cross_scene_estimator import (
    opening_candidates_from_intervals,
    polygon_area,
    polygon_validity,
    wall_endpoints,
)
from utils.joint_layout import build_joint_layout, door_endpoints
from utils.writer import xyz2json


def _rectangle(half_width=2.0, half_depth=1.25):
    return np.asarray(
        [
            [-half_width, -half_depth],
            [half_width, -half_depth],
            [half_width, half_depth],
            [-half_width, half_depth],
        ],
        dtype=np.float64,
    )


def _ray_depths(polygon, count=256):
    """Sample centered ray/segment intersections in the model's Bi x/z frame."""
    points = np.asarray(polygon, dtype=np.float64)
    longitude = (
        (np.arange(count, dtype=np.float64) + 0.5) / float(count) - 0.5
    ) * 2.0 * np.pi
    directions = np.stack([np.sin(longitude), np.cos(longitude)], axis=-1)
    depths = np.empty(count, dtype=np.float64)
    for token, direction in enumerate(directions):
        hits = []
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            edge = end - start
            denominator = direction[0] * edge[1] - direction[1] * edge[0]
            if abs(denominator) < 1e-12:
                continue
            distance = (start[0] * edge[1] - start[1] * edge[0]) / denominator
            ratio = (start[0] * direction[1] - start[1] * direction[0]) / denominator
            if distance > 1e-9 and -1e-9 <= ratio <= 1.0 + 1e-9:
                hits.append(distance)
        if not hits:
            raise AssertionError(f"ray {token} did not hit the synthetic polygon")
        depths[token] = min(hits)
    return depths


def _modulo_ninety_error(first, second):
    return abs((float(first) - float(second) + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)


def _wall_midpoint(layout, wall_index):
    start, end = wall_endpoints(layout, wall_index)
    return (start + end) / 2.0


def _signed_area(points):
    value = np.asarray(points, dtype=np.float64)
    return 0.5 * float(
        np.sum(
            value[:, 0] * np.roll(value[:, 1], -1)
            - np.roll(value[:, 0], -1) * value[:, 1]
        )
    )


class PredictedHeadingTest(unittest.TestCase):
    def test_arbitrary_rotations_and_winding(self):
        base = _rectangle()
        for degrees in (17.0, 43.0, -78.0):
            expected = np.deg2rad(degrees)
            rotated = rotate_floor_xz(base, expected)
            for points in (rotated, rotated[::-1]):
                heading, report = estimate_manhattan_heading(points)
                self.assertLess(_modulo_ninety_error(heading, expected), 1e-10)
                self.assertGreater(report["confidence"], 0.999999)
                self.assertEqual(
                    report["headingSource"],
                    "predicted_pre_manhattan_edges_mod90_length_weighted",
                )

    def test_heading_api_has_no_ground_truth_input(self):
        for function in (estimate_manhattan_heading, frame_preserving_manhattan):
            names = set(inspect.signature(function).parameters)
            self.assertFalse(
                names & {"gt", "ground_truth", "gt_heading", "gt_polygon", "gt_pose"}
            )

    def test_short_edges_and_small_prediction_noise_do_not_dominate(self):
        base = np.asarray(
            [
                [-2.0, -1.25],
                [-0.01, -1.25],
                [0.0, -1.249],
                [2.0, -1.25],
                [2.0, 1.25],
                [-2.0, 1.25],
            ],
            dtype=np.float64,
        )
        rng = np.random.RandomState(13)
        expected = np.deg2rad(31.0)
        noisy = rotate_floor_xz(base, expected) + rng.normal(0.0, 0.002, base.shape)
        heading, report = estimate_manhattan_heading(noisy)
        self.assertLess(_modulo_ninety_error(heading, expected), np.deg2rad(0.5))
        self.assertGreater(report["confidence"], 0.99)


class FramePreservingFitTest(unittest.TestCase):
    def test_canonical_fit_inverse_rotation_preserves_frame_and_area(self):
        for degrees in (17.0, 43.0, -78.0):
            expected = np.deg2rad(degrees)
            polygon = rotate_floor_xz(_rectangle(), expected)
            depth = _ray_depths(polygon)
            outputs, reports = frame_preserving_manhattan(depth)
            output = np.asarray(outputs[0], dtype=np.float64)
            report = reports[0]
            canonical = np.asarray(
                report["layoutPolygonManhattanCanonicalBiXZ"], dtype=np.float64
            )
            local = np.asarray(
                report["layoutPolygonPanoramaLocalBiXZ"], dtype=np.float64
            )

            fitted_heading, _ = estimate_manhattan_heading(local)
            self.assertLess(_modulo_ninety_error(fitted_heading, expected), np.deg2rad(1.0))
            canonical_heading, _ = estimate_manhattan_heading(canonical)
            self.assertLess(_modulo_ninety_error(canonical_heading, 0.0), np.deg2rad(0.25))
            np.testing.assert_allclose(output[:, ::2], local, atol=1e-12, rtol=0.0)
            self.assertAlmostEqual(polygon_area(canonical), polygon_area(local), places=10)
            self.assertGreater(_signed_area(canonical) * _signed_area(local), 0.0)
            self.assertLess(report["roundtripMaxAbsError"], 1e-12)
            self.assertLess(report["canonicalLocalAreaAbsDifference"], 1e-10)
            self.assertTrue(polygon_validity(canonical)["valid"])
            self.assertTrue(polygon_validity(local)["valid"])
            self.assertTrue(np.isfinite(local).all())

    def test_wrap_interval_and_opposite_walls_use_inverse_rotated_polygon(self):
        depth = _ray_depths(rotate_floor_xz(_rectangle(), np.deg2rad(17.0)))
        layout, report = prediction_to_layout(
            torch.as_tensor(depth, dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            layout_frame_mode="panorama_preserving",
        )
        probability = np.full(256, 0.1, dtype=np.float64)
        probability[[254, 255, 0, 1, 2, 126, 127, 128, 129, 130]] = 0.9
        candidates = opening_candidates_from_intervals(
            layout, [(254, 2), (126, 130)], probability
        )

        self.assertFalse(report["fallbackUsed"])
        self.assertEqual(report["layoutFrame"], "panorama_local")
        self.assertFalse(report["notValidForCrossScenePose"])
        self.assertEqual([value.candidate_index for value in candidates], [0, 1])
        self.assertEqual(
            [(value.token_start, value.token_end) for value in candidates],
            [(254, 2), (126, 130)],
        )
        self.assertNotEqual(candidates[0].wall_index, candidates[1].wall_index)
        midpoint_a = _wall_midpoint(layout, candidates[0].wall_index)
        midpoint_b = _wall_midpoint(layout, candidates[1].wall_index)
        self.assertLess(float(np.dot(midpoint_a, midpoint_b)), 0.0)
        for candidate in candidates:
            endpoints = door_endpoints(layout, candidate.spec)
            wall_start, wall_end = wall_endpoints(layout, candidate.wall_index)
            wall_vector = wall_end - wall_start
            cross = np.cross(wall_vector, endpoints - wall_start)
            np.testing.assert_allclose(cross, 0.0, atol=1e-8)

    def test_known_arbitrary_yaw_pair_retains_b_to_a_direction(self):
        depths = [
            _ray_depths(rotate_floor_xz(_rectangle(), np.deg2rad(value)))
            for value in (17.0, -26.0)
        ]
        layouts = [
            prediction_to_layout(
                torch.as_tensor(depth, dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
                layout_frame_mode="panorama_preserving",
            )[0]
            for depth in depths
        ]
        probability = np.ones(256, dtype=np.float64) * 0.5
        openings = [
            opening_candidates_from_intervals(layout, [interval], probability)[0]
            for layout, interval in zip(layouts, ((60, 66), (188, 194)))
        ]
        joint = build_joint_layout(
            layouts[0], layouts[1], openings[0].spec, openings[1].spec
        )
        transform_b_to_a = np.asarray(
            joint["alignment"]["roomBToWorld"], dtype=np.float64
        )
        yaw = math.atan2(transform_b_to_a[1, 0], transform_b_to_a[0, 0])
        distance_to_quarter_turn = min(
            abs((yaw - k * np.pi / 2.0 + np.pi) % (2.0 * np.pi) - np.pi)
            for k in range(-2, 3)
        )
        self.assertGreater(distance_to_quarter_turn, np.deg2rad(5.0))

        endpoints_b = door_endpoints(layouts[1], openings[1].spec)
        homogeneous_b = np.concatenate(
            [endpoints_b, np.ones((2, 1), dtype=np.float64)], axis=1
        )
        transformed_b = (transform_b_to_a @ homogeneous_b.T).T[:, :2]
        endpoints_a = door_endpoints(layouts[0], openings[0].spec)
        direct_error = np.max(np.abs(transformed_b - endpoints_a))
        reversed_error = np.max(np.abs(transformed_b - endpoints_a[::-1]))
        self.assertLess(min(direct_error, reversed_error), 1e-8)


class FrameModeCompatibilityTest(unittest.TestCase):
    def test_legacy_mode_exactly_reuses_historical_geometry(self):
        depth = _ray_depths(rotate_floor_xz(_rectangle(), np.deg2rad(17.0)))
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32)
        ratio_tensor = torch.tensor([1.25], dtype=torch.float32)
        safe_depth = np.maximum(
            np.abs(depth_tensor.detach().cpu().numpy().reshape(1, -1)), 1e-3
        )
        reference_xyz = post_process(safe_depth, type_name="manhattan")[0]
        reference_layout = xyz2json(reference_xyz, 1.25)

        default_layout, default_report = prediction_to_layout(depth_tensor, ratio_tensor)
        explicit_layout, explicit_report = prediction_to_layout(
            depth_tensor, ratio_tensor, layout_frame_mode="legacy_fixed_axis"
        )

        self.assertEqual(default_layout, explicit_layout)
        self.assertEqual(default_layout, reference_layout)
        self.assertEqual(default_report["layoutFrame"], "manhattan_canonical")
        self.assertTrue(default_report["notValidForCrossScenePose"])
        self.assertIsNone(default_report["layoutPolygonPanoramaLocalBiXZ"])
        self.assertEqual(explicit_report["layoutFrameMode"], "legacy_fixed_axis")

    def test_group_cli_is_opt_in_and_default_compatible(self):
        self.assertEqual(parse_args([]).layout_frame_mode, "legacy_fixed_axis")
        self.assertEqual(
            parse_args(["--layout_frame_mode", "panorama_preserving"]).layout_frame_mode,
            "panorama_preserving",
        )


if __name__ == "__main__":
    unittest.main()
