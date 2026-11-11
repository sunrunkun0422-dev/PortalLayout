import unittest
import numpy as np

from portallayout.geometry.joint_layout import generate_opening_pose_hypotheses
from portallayout.engine.opening_pair_pipeline import connected_components


class PoseHypothesesTest(unittest.TestCase):
    def test_identity_not_discarded_without_evidence(self):
        segment = [[0, 0], [2, 0]]
        result = generate_opening_pose_hypotheses(segment, segment)
        self.assertEqual(result['status'], 'unverified')
        self.assertEqual(len(result['hypotheses']), 2)
        self.assertTrue(any(np.allclose(h['T_B_to_A'], np.eye(3)) for h in result['hypotheses']))

    def test_endpoint_flip_preserves_hypothesis_set(self):
        a, b = [[0, 0], [2, 0]], [[1, 1], [2, 1]]
        x = generate_opening_pose_hypotheses(a, b)['hypotheses']
        y = generate_opening_pose_hypotheses(a[::-1], b)['hypotheses']
        for h in x: self.assertTrue(any(np.allclose(h['T_B_to_A'], k['T_B_to_A']) for k in y))
        self.assertEqual(x[0]['scale_B_to_A'], 2.)

    def test_independent_opening_resolves_identity_only_when_calibrated(self):
        a = [[0, 0], [2, 0]]
        evidence = [{'endpoints_a': [[0, 2], [2, 2]], 'endpoints_b': [[0, 2], [2, 2]],
                     'candidate_id_a': 'other_a', 'candidate_id_b': 'other_b'}]
        for calibrated in (False, True):
            r = generate_opening_pose_hypotheses(a, a, independent_matches=evidence, calibrated=calibrated)
            self.assertEqual(r['status'], 'verified' if calibrated else 'unverified')
            self.assertEqual(sum(h['supported'] for h in r['hypotheses']), 1)
            good = [h for h in r['hypotheses'] if h['supported']][0]
            np.testing.assert_allclose(good['T_B_to_A'], np.eye(3), atol=1e-12)

    def test_seed_cannot_verify_itself(self):
        a = [[0, 0], [2, 0]]
        r = generate_opening_pose_hypotheses(a, a, independent_matches=[{'endpoints_a': a, 'endpoints_b': a}], calibrated=True)
        self.assertEqual(r['status'], 'unverified')
        self.assertFalse(r['hypotheses'][0]['independent_evidence'])

    def test_multiple_supported_hypotheses_remain_ambiguous(self):
        a = [[-1, 0], [1, 0]]
        holdout = [{'endpoints_a': [[0, -1], [0, 1]], 'endpoints_b': [[0, -1], [0, 1]]}]
        r = generate_opening_pose_hypotheses(a, a, independent_matches=holdout, calibrated=True)
        self.assertEqual(r['status'], 'ambiguous')

    def test_degenerate_and_nonfinite_rejected(self):
        for a in ([[0, 0], [0, 0]], [[0, 0], [float('nan'), 1]]):
            with self.assertRaises(ValueError): generate_opening_pose_hypotheses(a, [[0, 0], [1, 0]])

    def test_components_keep_isolates_and_do_not_force_connect(self):
        self.assertEqual(connected_components(['a', 'b', 'c'], [('a', 'b')]), [['a', 'b'], ['c']])


if __name__ == '__main__': unittest.main()
