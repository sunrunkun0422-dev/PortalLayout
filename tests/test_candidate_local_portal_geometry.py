import csv
import inspect
import json
import math
import unittest
from pathlib import Path

import numpy as np

from tools.audit_zind_portal_geometry import (
    _room_predicted_inputs,
    load_inputs,
    parse_args as parse_audit_args,
)
from utils.candidate_local_portal_geometry import (
    CandidateLocalPortalConfig,
    bi_to_zind,
    estimate_candidate_local_portal_geometry,
    panorama_ray,
)
from utils.joint_layout import build_joint_layout_from_endpoints


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (
    REPO_ROOT
    / "src/output/zind_room_group_debug/t3_portal_geometry_audit_0665_floor03"
)


def _circle_contour(count=256, radius=2.5):
    return np.stack([panorama_ray(index, count) * radius for index in range(count)])


def _put_ray_line(contour, start, end, normal, offset):
    count = len(contour)
    index = start
    while True:
        ray = panorama_ray(index, count)
        contour[index] = ray * (offset / float(np.dot(normal, ray)))
        if index == end:
            break
        index = (index + 1) % count


class CandidateLocalPortalProductionTest(unittest.TestCase):
    def test_known_local_wall_and_endpoint_order(self):
        contour = _circle_contour()
        normal = np.asarray([0.0, 1.0])
        _put_ray_line(contour, 108, 148, normal, 2.0)
        result = estimate_candidate_local_portal_geometry(
            contour, (112, 144), 256, np.ones(256)
        )
        self.assertTrue(result.geometry_valid)
        self.assertEqual(result.frame, "panorama_local")
        expected = np.stack([
            panorama_ray(112) * (2.0 / panorama_ray(112)[1]),
            panorama_ray(144) * (2.0 / panorama_ray(144)[1]),
        ])
        np.testing.assert_allclose(
            result.primary.endpoints_local_bi_xz, expected, atol=1e-10
        )

    def test_wraparound_interval(self):
        contour = _circle_contour()
        normal = np.asarray([0.0, 1.0])
        _put_ray_line(contour, 246, 9, normal, -2.0)
        result = estimate_candidate_local_portal_geometry(
            contour, (250, 5), 256, np.full(256, 0.8)
        )
        self.assertTrue(result.geometry_valid)
        endpoints = np.asarray(result.primary.endpoints_local_bi_xz)
        np.testing.assert_allclose(endpoints[:, 1], -2.0, atol=1e-10)

    def test_noisy_wall_with_outlier_keeps_predicted_only_result(self):
        contour = _circle_contour()
        normal = np.asarray([0.0, 1.0])
        _put_ray_line(contour, 105, 151, normal, 2.0)
        rng = np.random.RandomState(31)
        contour[105:152] += rng.normal(0.0, 0.004, (47, 2))
        contour[121] += np.asarray([0.25, -0.18])
        result = estimate_candidate_local_portal_geometry(
            contour, (109, 147), 256, np.linspace(0.2, 0.9, 256)
        )
        self.assertTrue(result.geometry_valid)
        self.assertEqual(result.primary_variant, "ransac")
        self.assertLess(result.primary.quality["normalized_residual"], 0.02)
        self.assertFalse(result.to_json()["gt_available_to_geometry"])

    def test_corner_neighborhood_is_deterministic_and_reports_quality(self):
        contour = _circle_contour()
        _put_ray_line(contour, 104, 128, np.asarray([0.0, 1.0]), 2.0)
        _put_ray_line(contour, 129, 152, np.asarray([1.0, 0.0]), 0.8)
        first = estimate_candidate_local_portal_geometry(
            contour, (112, 144), 256, np.ones(256)
        )
        second = estimate_candidate_local_portal_geometry(
            contour, (112, 144), 256, np.ones(256)
        )
        self.assertEqual(first.to_json(), second.to_json())
        self.assertIn("angular_dispersion_degrees", first.primary.quality)

    def test_nonfinite_and_degenerate_inputs_return_explicit_failure(self):
        contour = _circle_contour()
        contour[3, 0] = float("nan")
        invalid = estimate_candidate_local_portal_geometry(
            contour, (10, 20), 256, np.ones(256)
        )
        self.assertFalse(invalid.geometry_valid)
        self.assertIn("non-finite", invalid.failure_reason)

        zero = estimate_candidate_local_portal_geometry(
            np.zeros((256, 2)), (10, 20), 256, np.ones(256)
        )
        self.assertFalse(zero.geometry_valid)
        self.assertIn("radius", zero.failure_reason)

        missing_interval = estimate_candidate_local_portal_geometry(
            _circle_contour(), None, 256, np.ones(256)
        )
        self.assertFalse(missing_interval.geometry_valid)
        self.assertIn("start and end", missing_interval.failure_reason)

    def test_formal_signature_has_no_gt(self):
        names = set(inspect.signature(estimate_candidate_local_portal_geometry).parameters)
        self.assertFalse(
            names & {"gt", "ground_truth", "gt_endpoints", "gt_polygon", "gt_pose"}
        )


class FrozenAuditConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = parse_audit_args([])
        required = (
            Path(args.rooms_json), Path(args.pairs_json), Path(args.group_manifest),
            AUDIT_ROOT / "audit_summary.json", AUDIT_ROOT / "portal_hypotheses.csv",
        )
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("requires local frozen T3 inputs and portal audit results")
        cls.inputs = load_inputs(args)
        cls.rooms = _room_predicted_inputs(cls.inputs["rooms"])
        cls.summary = json.loads((AUDIT_ROOT / "audit_summary.json").read_text())
        with (AUDIT_ROOT / "portal_hypotheses.csv").open() as file:
            cls.rows = list(csv.DictReader(file))

    def _estimate(self, room, candidate_index):
        state = self.rooms[room]
        candidate = next(
            value
            for value in state["opening_prediction"]["candidates"]
            if int(value["candidateIndex"]) == int(candidate_index)
        )
        return estimate_candidate_local_portal_geometry(
            state["pre_contour_bi_xz"],
            (candidate["tokenStart"], candidate["tokenEnd"]),
            256,
            state["opening_prediction"]["probability"],
            CandidateLocalPortalConfig(),
        )

    def test_formal_m2_endpoints_match_frozen_audit_exactly(self):
        for pair_id, pair in self.inputs["pairs"].items():
            if not pair["is_match"]:
                continue
            for side, room, index in (
                ("A", pair["room_a"], pair["best_candidate_pair"][0]),
                ("B", pair["room_b"], pair["best_candidate_pair"][1]),
            ):
                result = self._estimate(room, index)
                frozen = next(
                    row
                    for row in self.rows
                    if row["pair_id"] == pair_id
                    and row["side"] == side
                    and row["method"] == "M2"
                    and row["selected_predicted_only"] == "True"
                )
                self.assertEqual(result.primary_variant, frozen["variant"])
                np.testing.assert_allclose(
                    result.primary.endpoints_local_zind_xy,
                    json.loads(frozen["endpoints_zind_local_xy"]),
                    atol=1e-12,
                    rtol=0.0,
                )

    def test_formal_joint_pose_matches_frozen_audit_exactly(self):
        for pair_id, pair in self.inputs["pairs"].items():
            if not pair["is_match"]:
                continue
            room_a, room_b = pair["room_a"], pair["room_b"]
            result_a = self._estimate(room_a, pair["best_candidate_pair"][0])
            result_b = self._estimate(room_b, pair["best_candidate_pair"][1])
            joint = build_joint_layout_from_endpoints(
                self.rooms[room_a]["frame_layout"],
                self.rooms[room_b]["frame_layout"],
                result_a.primary.endpoints_local_zind_xy,
                result_b.primary.endpoints_local_zind_xy,
                boundary_a=bi_to_zind(self.rooms[room_a]["pre_contour_bi_xz"]),
                boundary_b=bi_to_zind(self.rooms[room_b]["pre_contour_bi_xz"]),
            )
            expected = self.summary["method_comparison"][pair_id]["M2"]["pose"]
            np.testing.assert_allclose(
                joint["alignment"]["roomBToWorld"],
                expected["T_B_to_A"],
                atol=1e-12,
                rtol=0.0,
            )
            matrix = np.asarray(joint["alignment"]["roomBToWorld"])
            self.assertAlmostEqual(
                math.hypot(matrix[0, 0], matrix[1, 0]),
                expected["relative_scale"],
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
