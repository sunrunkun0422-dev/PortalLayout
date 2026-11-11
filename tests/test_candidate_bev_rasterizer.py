import unittest

import numpy as np
import torch

from portallayout.utils.candidate_bev_rasterizer import (
    CandidateBEVRasterConfig,
    normalize_portal_frame,
    rasterize_candidate_bev,
)


class CandidateBEVRasterizerTest(unittest.TestCase):
    def setUp(self):
        self.a = [[-2, 0], [2, 0], [2, 3], [-2, 3]]
        self.b = [[-2, -3], [2, -3], [2, 0], [-2, 0]]
        self.portal = [[-0.5, 0], [0.5, 0]]

    def test_portal_center_direction_and_scale(self):
        normalized, metadata = normalize_portal_frame(self.portal, self.portal)
        np.testing.assert_allclose(normalized, [[-0.5, 0.0], [0.5, 0.0]], atol=1e-12)
        self.assertEqual(metadata["portal_width_A"], 1.0)

    def test_seven_channels_are_finite_and_semantic(self):
        tensor, metadata = rasterize_candidate_bev(self.a, self.b, self.portal)
        self.assertEqual(tuple(tensor.shape), (7, 128, 128))
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertTrue(metadata["valid"])
        self.assertGreater(float(tensor[0].sum()), 0.0)
        self.assertGreater(float(tensor[1].sum()), 0.0)
        self.assertGreater(float(tensor[5].sum()), 0.0)
        self.assertTrue(torch.all(tensor[6] == 1))

    def test_overlap_and_ab_exchange_have_symmetric_mass(self):
        overlap_b = [[-1, 1], [3, 1], [3, 4], [-1, 4]]
        first, _ = rasterize_candidate_bev(self.a, overlap_b, self.portal)
        second, _ = rasterize_candidate_bev(overlap_b, self.a, self.portal)
        self.assertGreater(float(first[4].sum()), 0.0)
        self.assertAlmostEqual(float(first[4].sum()), float(second[4].sum()), places=5)
        self.assertAlmostEqual(float(first[0].sum()), float(second[1].sum()), places=5)
        self.assertAlmostEqual(float(first[1].sum()), float(second[0].sum()), places=5)

    def test_invalid_geometry_is_explicit_zero_with_mask(self):
        tensor, metadata = rasterize_candidate_bev(
            self.a, self.b, self.portal, geometry_valid=False,
            config=CandidateBEVRasterConfig(grid_size=64),
        )
        self.assertEqual(tuple(tensor.shape), (7, 64, 64))
        self.assertEqual(float(tensor.sum()), 0.0)
        self.assertFalse(metadata["valid"])
        self.assertEqual(metadata["failure_reason"], "geometry_valid=false")


if __name__ == "__main__":
    unittest.main()
