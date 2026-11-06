"""Contracts for the isolated GT-opening oracle (synthetic data only)."""
import math
import unittest

import numpy as np

from tools.eval_zind_gt_opening_pose_oracle import assess, solve_pose
from tools.evaluate_zind_room_group import _pose


class OraclePoseContractTest(unittest.TestCase):
    def test_production_solver_and_endpoint_order_invariance(self):
        a = np.array([[0., 0.], [2., 0.]])
        b = np.array([[1., 1.], [2., 1.]])
        ba = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
        bb = np.array([[1., 1.], [2., 1.], [2., 0.], [1., 0.]])
        expected = np.array([[2., 0., -2.], [0., 2., -2.], [0., 0., 1.]])
        for ea, eb in [(a, b), (a[::-1], b), (a, b[::-1]), (a[::-1], b[::-1])]:
            pose, alignment = solve_pose(ea, eb, ba, bb)
            np.testing.assert_allclose(pose['T_B_to_A'], expected, atol=1e-12)
            self.assertEqual(alignment['transformDirection'], 'B_to_A')

    def test_existing_evaluator_wraps_yaw_and_converts_units_once(self):
        def pose(deg, tx, scale):
            theta = math.radians(deg)
            return _pose(np.array([[scale*math.cos(theta), -scale*math.sin(theta), tx],
                                   [scale*math.sin(theta), scale*math.cos(theta), 0.],
                                   [0., 0., 1.]]))
        result = assess(pose(359, 2, 2), pose(1, 1, 1), 3.)
        self.assertAlmostEqual(result['yaw_error_degrees'], 2.)
        self.assertAlmostEqual(result['translation_error_meters'], 3.)
        self.assertAlmostEqual(result['absolute_log_scale_error'], math.log(2))

    def test_degenerate_gt_opening_fails_instead_of_inventing_pose(self):
        boundary = [[0, 0], [1, 0], [0, 1]]
        with self.assertRaises(ValueError):
            solve_pose([[0, 0], [0, 0]], [[0, 0], [1, 0]], boundary, boundary)
