import copy
import unittest
from tools.research10_r06_review import select_candidates, validate_cohort, projection_index
from tools.research10_r06_review_status import count_review


class TestReviewSampling(unittest.TestCase):
    def test_round_robin_is_deterministic_and_preserves_house_coverage(self):
        rows = [dict(floor_uid='floor_'+str(h), pair_id='pair_'+str(n), house_id=str(h))
                for h in range(4) for n in range(5)]
        a = select_candidates(rows, 123, 9)
        b = select_candidates(list(reversed(rows)), 123, 9)
        self.assertEqual(a, b)
        self.assertEqual(len({r['house_id'] for r in a[:4]}), 4)
        self.assertEqual(len(a), 9)

    def test_duplicate_pair_is_rejected(self):
        r = dict(floor_uid='floor', pair_id='pair', house_id='house')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            select_candidates([r, r.copy()], 123, 2)

    def test_cohort_rejects_test_data_and_role_leakage(self):
        roles = dict(fit=['1'], selection=['2'], calibration=['3'], audit=['4'])
        rows = [dict(floor_uid='floor_'+x, house_id=x, role=role) for role, hs in roles.items() for x in hs]
        samples = {r['floor_uid']: dict(house_id=r['house_id'], official_split='train') for r in rows}
        cohort = dict(roles=roles, cohort=rows)
        self.assertEqual(validate_cohort(cohort, samples), rows)
        altered = copy.deepcopy(samples); altered['floor_4']['official_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'official train'):
            validate_cohort(cohort, altered)
        altered = copy.deepcopy(cohort); altered['roles']['audit'].append('1')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_cohort(altered, samples)

    def test_projection_index_excludes_missing_and_retains_seam(self):
        row = {'interfaces_by_layout': {'layout_raw': [
            {'interface_observation_id': 'good', 'projection': dict(endpoint_u=[.98, .02], top_v=[.4, .3], bottom_v=[.8, .7], wraps_seam=True)},
            {'interface_observation_id': 'bad', 'projection': {}}
        ]}}
        result = projection_index([row])
        self.assertEqual(list(result), ['good'])
        self.assertTrue(result['good']['wraps_seam'])
        self.assertEqual(result['good']['endpoint_u'], [.98, .02])

    def test_unknown_is_pending_and_cannot_be_promoted_to_available(self):
        pack = dict(pack_id='pack', cases=[dict(case_id='one', batch='initial', house_id='001', role='fit')])
        self.assertEqual(count_review(pack)['batches']['initial']['counts']['direction_pending'], 2)
        blank = dict(status='', occlusion='', belongs='', points=[])
        review = dict(schema='r06-human-review.v1', pack_id='pack', reviewer='reviewer', exported_at='2026-01-01',
                      cases=[dict(case_id='one', shared='uncertain', a_to_b=dict(blank, status='available'), b_to_a=blank)])
        with self.assertRaisesRegex(ValueError, 'available direction'):
            count_review(pack, review)
        review['cases'][0]['shared'] = 'yes'
        review['cases'][0]['a_to_b'] = dict(status='available', occlusion='clear', belongs='yes', points=[[.1, .2], [.3, .4]])
        result = count_review(pack, review)['batches']['initial']
        self.assertEqual(result['counts']['direction_available'], 1)
        self.assertEqual(result['available_houses'], 1)

    def test_review_pack_identity_and_points_are_checked(self):
        pack = dict(pack_id='pack', cases=[dict(case_id='one', batch='initial', house_id='001', role='fit')])
        blank = dict(status='', occlusion='', belongs='', points=[])
        review = dict(schema='r06-human-review.v1', pack_id='wrong', reviewer='r', exported_at='date',
                      cases=[dict(case_id='one', shared='yes', a_to_b=blank, b_to_a=blank)])
        with self.assertRaisesRegex(ValueError, 'identity'):
            count_review(pack, review)
        review['pack_id'] = 'pack'
        review['cases'][0]['a_to_b'] = dict(blank, points=[[1.2, 0]])
        with self.assertRaisesRegex(ValueError, 'points'):
            count_review(pack, review)

if __name__ == '__main__':
    unittest.main()
