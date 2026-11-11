import copy
import math

import numpy as np
import pytest

from tools.research10_matching_baselines import score_fixed_candidates


def candidate(uid, appearance=(1., 0.), valid_mask=(True, True), width=1., width_valid=True, valid=True):
    return {'candidate_id': uid, 'valid': valid, 'appearance': list(appearance),
            'appearance_valid': list(valid_mask), 'opening_width': width, 'width_valid': width_valid}


def view(uid, candidates, factor=1., unit='meters', source='measured_camera_height', scale_valid=True):
    return {'view_id': uid, 'appearance_source': 'predicted_single_view', 'width_source': 'predicted_single_view',
            'scale': {'valid': scale_valid, 'factor_to_common': factor, 'common_unit': unit, 'source': source},
            'candidates': candidates}


def data(a, b, pairs):
    return {'schema': 'r04-fixed-candidates.v1', 'view_a': a, 'view_b': b,
            'candidate_pairs': [{'candidate_id_a': i, 'candidate_id_b': j} for i, j in pairs]}


def test_fixed_pairs_appearance_and_physical_width_not_prediction_decisions():
    a = view('A', [candidate('a0', (1, 0), width=.5), candidate('a1', (0, 1), width=1.)], factor=2.)
    b = view('B', [candidate('b0', (1, 0), width=1.), candidate('b1', (0, 1), width=2.)])
    result = score_fixed_candidates(data(a, b, [('a1', 'b0'), ('a0', 'b1'), ('a0', 'b0')]))
    assert result['pair_count'] == 3
    assert [(r['candidate_id_a'], r['candidate_id_b']) for r in result['rows']] == [('a1', 'b0'), ('a0', 'b1'), ('a0', 'b0')]
    assert [r['appearance_cosine']['score'] for r in result['rows']] == [0., 0., 1.]
    np.testing.assert_allclose([r['width_ratio']['score'] for r in result['rows']], [.5, .5, 1.])
    assert result['calibration'] == 'not_calibrated' and result['scoring_only']
    assert not {'matches', 'accepted', 'probability', 'threshold'} & result.keys()


def test_reordering_candidates_does_not_reindex_fixed_ids():
    a = view('A', [candidate('x', (1, 0)), candidate('y', (0, 1))])
    b = view('B', [candidate('z', (0, 1)), candidate('w', (1, 0))])
    spec = data(a, b, [('x', 'z'), ('y', 'z'), ('x', 'missing')])
    result = score_fixed_candidates(spec)
    changed = copy.deepcopy(spec)
    changed['view_a']['candidates'].reverse(); changed['view_b']['candidates'].reverse()
    assert score_fixed_candidates(changed) == result
    assert result['rows'][2]['appearance_cosine']['reason'] == 'missing_candidate_b'
    assert result['rows'][2]['width_ratio']['reason'] == 'missing_candidate_b'


def test_invalid_feature_dimensions_do_not_become_zero_or_a_match():
    a = view('A', [candidate('a0', (1., 999.), (True, False)),
                   candidate('a1', (0., 0.)), candidate('a2', (1., 2., 3.), (True, True, True)),
                   candidate('a3', (float('nan'), 1.))])
    b = view('B', [candidate('b')])
    result = score_fixed_candidates(data(a, b, [(f'a{i}', 'b') for i in range(4)]))
    assert [r['appearance_cosine']['reason'] for r in result['rows']] == [
        'a_appearance_values_unknown', 'a_zero_appearance_norm',
        'appearance_dimensions_mismatch', 'a_appearance_values_unknown']
    assert all(r['appearance_cosine']['score'] is None for r in result['rows'])
    assert all(r['width_ratio']['available'] for r in result['rows'])


def test_masked_width_and_unknown_or_incompatible_scale_stay_unknown():
    a = view('A', [candidate('a', width_valid=False)])
    b = view('B', [candidate('b')])
    assert score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]['width_ratio']['reason'] == 'a_width_unknown'
    a['candidates'][0]['width_valid'] = True
    a['scale']['valid'] = False; a['scale']['factor_to_common'] = None; a['scale']['source'] = 'unavailable'
    assert score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]['width_ratio']['reason'] == 'a_scale_unknown'
    a['scale'] = dict(valid=True, factor_to_common=1., common_unit='shared_camera_height', source='shared_height_assumption')
    assert score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]['width_ratio']['reason'] == 'incompatible_common_units'
    b['scale'] = dict(valid=True, factor_to_common=1., common_unit='shared_camera_height', source='predicted_scale')
    assert score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]['width_ratio']['reason'] == 'incompatible_shared_height_provenance'
    b['scale']['source'] = 'shared_height_assumption'
    assert score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]['width_ratio']['score'] == 1.


def test_width_similarity_invariant_to_rigid_transforms_and_scaled_units():
    p = np.array([[0., 0.], [1.2, .5]])
    theta = .83; R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    transformed = p @ R.T + np.array([51., -23.])
    physical_width = float(np.linalg.norm(p[1]-p[0]))
    measured = float(np.linalg.norm(transformed[1]-transformed[0]))
    assert math.isclose(physical_width, measured, abs_tol=1e-12)
    a = view('A', [candidate('a', width=physical_width/1.5)], factor=1.5)
    b = view('B', [candidate('b', width=measured/1.8)], factor=1.8)
    result = score_fixed_candidates(data(a, b, [('a', 'b')]))
    assert math.isclose(result['rows'][0]['width_ratio']['score'], 1., abs_tol=1e-12)


def test_empty_invalid_and_duplicate_candidate_cases():
    a, b = view('A', []), view('B', [])
    assert score_fixed_candidates(data(a, b, []))['rows'] == []
    assert score_fixed_candidates(data(a, b, [('x', 'y')]))['rows'][0]['width_ratio']['reason'] == 'missing_candidate_a'
    a['candidates'] = [candidate('x', valid=False)]
    assert score_fixed_candidates(data(a, view('B', [candidate('y')]), [('x', 'y')]))['rows'][0]['appearance_cosine']['reason'] == 'invalid_candidate_a'
    a['candidates'].append(candidate('x'))
    with pytest.raises(ValueError, match='duplicate candidate'):
        score_fixed_candidates(data(a, b, []))
    a['candidates'].pop()
    with pytest.raises(ValueError, match='duplicate fixed'):
        score_fixed_candidates(data(a, b, [('x', 'y'), ('x', 'y')]))


def test_unapproved_gt_and_pose_fields_are_rejected():
    spec = data(view('A', [candidate('a')]), view('B', [candidate('b')]), [('a', 'b')])
    bad = copy.deepcopy(spec); bad['view_a']['T_B_to_A_gt'] = np.eye(3).tolist()
    with pytest.raises(ValueError, match='only explicit prediction fields'):
        score_fixed_candidates(bad)
    bad = copy.deepcopy(spec); bad['view_a']['candidates'][0]['target'] = 1
    with pytest.raises(ValueError, match='only prediction fields'):
        score_fixed_candidates(bad)
    bad = copy.deepcopy(spec); bad['view_a']['scale']['source'] = 'ground_truth'
    with pytest.raises(ValueError, match='disallowed scale source'):
        score_fixed_candidates(bad)


def test_large_features_and_extreme_widths_do_not_overflow():
    a = view('A', [candidate('a', (1e300, 1e300), width=1e-200)])
    b = view('B', [candidate('b', (1e300, 1e300), width=1e200)])
    result = score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]
    assert math.isclose(result['appearance_cosine']['score'], 1.)
    assert result['width_ratio']['score'] == 0.


def test_malformed_masks_and_string_geometry_do_not_become_valid_numbers():
    a = view('A', [candidate('a', (1, 2), (True,), width='1.0')])
    b = view('B', [candidate('b')])
    row = score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]
    assert row['appearance_cosine']['reason'] == 'a_invalid_appearance_dimensions'
    assert row['width_ratio']['reason'] == 'a_invalid_width'
    a['candidates'][0] = candidate('a', ('1', '0'))
    row = score_fixed_candidates(data(a, b, [('a', 'b')]))['rows'][0]
    assert row['appearance_cosine']['reason'] == 'a_appearance_values_unknown'
