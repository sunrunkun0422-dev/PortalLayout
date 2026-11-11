"""Evaluation must not hide unretrieved positives or relabel unknown matches."""
import unittest
from tools.report_opening_matches import summarize, join_predictions
from portallayout.utils.room_feature_cache import fingerprint


class OpeningMatchReportTests(unittest.TestCase):
    def test_recall_includes_unretrieved_and_unknown_is_not_negative(self):
        rows = [dict(house_id='h', positive_cells=3, known_negative_pair=False,
                     numerical_status='assigned', matches=[dict(target=1, confidence=.9),
                                                           dict(target=-1, confidence=.95)])]
        report = summarize(rows, .8)
        self.assertEqual(report['recall_all_weak_positives'], 1/3)
        self.assertEqual(report['precision_known'], 1.)
        self.assertEqual(report['accepted_unknown'], 1)
        self.assertEqual(report['unretrieved_positive_cells'], 2)

    def fixture(self):
        weak = dict(source_snapshot='s', pairs=[dict(pair_id='dataset-id', observation_a='a', observation_b='b',
                    floor_uid='f', house_id='h', candidate_ids_a=['ca'], candidate_ids_b=['cb'], weak_cells=[[1]])])
        auxiliary = dict(source_snapshot='s', house_protocol=dict(fit=['h'], selection=[], calibration=[]),
                         opening_pairs=[dict(pair_id=fingerprint(['a', 'b']), house_id='h', role='fit',
                         numerical_status='assigned', matches=[dict(index_a=0, index_b=0, candidate_id_a='ca',
                         candidate_id_b='cb', confidence=.8, target=1)])])
        gt = {'f': {'observations': {'a': dict(room_id='r1', membership_confirmed=True),
                                    'b': dict(room_id='r2', membership_confirmed=True)}}}
        return weak, auxiliary, gt

    def test_join_uses_observations_not_incompatible_pair_ids(self):
        rows = join_predictions(*self.fixture())
        self.assertEqual(rows[0]['relationship'], 'cross_room')
        self.assertEqual(summarize(rows, .5)['recall_all_weak_positives'], 1.)

    def test_missing_prediction_retains_denominator_and_identity_is_checked(self):
        weak, auxiliary, gt = self.fixture()
        auxiliary['opening_pairs'][0]['matches'][0]['candidate_id_b'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity'):
            join_predictions(weak, auxiliary, gt)
        auxiliary['opening_pairs'] = []
        rows = join_predictions(weak, auxiliary, gt)
        report = summarize(rows, 0.)
        self.assertEqual(report['recall_all_weak_positives'], 0.)
        self.assertEqual(report['numerical_status'], {'missing_prediction': 1})

    def test_negative_pair_false_acceptance_and_no_acceptance(self):
        rows = [dict(house_id='h', positive_cells=0, known_negative_pair=True,
                     numerical_status='assigned', matches=[dict(target=0, confidence=.4)])]
        self.assertEqual(summarize(rows, .3)['negative_pair_false_acceptance'], 1.)
        rejected = summarize(rows, .5)
        self.assertIsNone(rejected['precision_known'])
        self.assertIsNone(rejected['recall_all_weak_positives'])
        self.assertEqual(rejected['negative_pair_false_acceptance'], 0.)


if __name__ == '__main__':
    unittest.main()
