import copy
import unittest
import numpy as np

from portallayout.datasets.opening_pair_supervision import (circular_iou, associate_candidates, complete_targets, pair_cells)


class WeakSupervisionTest(unittest.TestCase):
    def fixture(self, intervals=((90, 9),), gt=((.9, .2),)):
        record = {'prediction': {'openings': [{'candidate_id': str(i), 'token_interval_inclusive': list(v)}
                                              for i, v in enumerate(intervals)]},
                  'tensor_spec': {'depth_enclosed': {'shape': [100]}}}
        annotation = {'layout_validity': {'layout_raw_interface_inventory': True},
            'interfaces_by_layout': {'layout_raw': [
                {'kind': 'door', 'interface_observation_id': 'gt'+str(i),
                 'projection': {'start_u': s, 'length_u': l}} for i, (s, l) in enumerate(gt)]}}
        return record, annotation

    def test_circular_iou_and_seam_association(self):
        self.assertAlmostEqual(circular_iou(.9, .2, -.1, .2), 1.)
        self.assertAlmostEqual(circular_iou(.9, .2, .95, .1), .5)
        rows, _ = associate_candidates(*self.fixture())
        self.assertEqual(rows[0]['interface_id'], 'gt0')

    def test_duplicate_predictions_are_all_ambiguous(self):
        rows, _ = associate_candidates(*self.fixture(intervals=((90, 9), (90, 9))))
        self.assertTrue(all(r['status'] == 'ambiguous' for r in rows))

    def test_merged_gt_is_ambiguous(self):
        rows, _ = associate_candidates(*self.fixture(intervals=((10, 39),), gt=((.1, .2), (.3, .1))))
        self.assertEqual(rows[0]['status'], 'ambiguous')

    def test_missing_annotation_is_not_false_positive_label(self):
        record, annotation = self.fixture(); annotation['layout_validity'] = {}
        rows, _ = associate_candidates(record, annotation)
        self.assertEqual(rows[0]['status'], 'missing_annotation')

    def test_complete_rows_only_and_explicit_unmatched(self):
        a, b = complete_targets(np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0]]))
        self.assertEqual(a, [0, -2, -1]); self.assertEqual(b, [-2, -1, -1])
        self.assertEqual(complete_targets(np.empty((0, 3), dtype=int)), ([], [-2]*3))

    def test_multiple_positives_are_ignored(self):
        self.assertEqual(complete_targets([[1, 1]])[0], [-2])

    def test_no_link_is_unknown_until_geometric_negative_evidence(self):
        pair = {'portal_correspondences': [], 'physical_portal_label': 'unknown', 'evidence': None}
        assoc = [{'status': 'associated', 'interface_id': 'gt'}]
        gt = {'gt': {'floor_endpoints': [[0, 0], [1, 0]]}}
        self.assertEqual(pair_cells(pair, assoc, assoc, gt, gt, 1).tolist(), [[-1]])
        other = {'gt': {'floor_endpoints': [[3, 0], [4, 0]]}}
        self.assertEqual(pair_cells(pair, assoc, assoc, gt, other, 1).tolist(), [[0]])
        self.assertEqual(pair_cells(pair, assoc, assoc, gt, other, None).tolist(), [[-1]])

    def test_positive_proxy_and_explicit_negative_certificate(self):
        assoc = [{'status': 'associated', 'interface_id': 'gt'}]
        pair = {'portal_correspondences': [{'source_interface_a': 'gt', 'source_interface_b': 'gt'}],
                'physical_portal_label': 'shared'}
        self.assertEqual(pair_cells(pair, assoc, assoc, {}, {}, None).tolist(), [[1]])
        pair.update(portal_correspondences=[], physical_portal_label='no_shared', evidence={'clearance': 1})
        self.assertEqual(pair_cells(pair, assoc, assoc, {}, {}, None).tolist(), [[0]])


if __name__ == '__main__': unittest.main()
