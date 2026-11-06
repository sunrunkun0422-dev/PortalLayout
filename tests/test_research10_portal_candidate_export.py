import copy

import pytest
from tools.research10_candidate_comparison import build_union, CANDIDATE_VERSION
from tools.research10_portal_candidate_export import export_prediction


def prediction():
    return {'schema_version': 'research10.geometry-prediction.v1', 'ground_truth_read': False,
            'status': 'predicted', 'floor_uid': 'f', 'pair_denominator': 3,
            'observations': [{'observation_id': x, 'prediction': {'coordinate_spec': {
                'frame': 'bi_panorama_local_xz', 'units': 'camera_height_normalized'}}} for x in 'abc'],
            'selected_pairs': [
                {'observation_a': a, 'observation_b': b, 'pose_hypotheses': [
                    {'hypothesis_id': 'h1', 'seed_candidate_ids': ['oa', 'ob'],
                     'T_B_to_A': [[1, 0, 2], [0, 1, 3], [0, 0, 1]]}]
                 if (a, b) == ('a', 'b') else []}
                for a, b in [('a', 'b'), ('a', 'c'), ('b', 'c')]]}


def test_export_preserves_all_pairs_and_native_frame():
    index, portal = export_prediction(prediction())
    assert len(index['pairs']) == 3
    assert len(portal['pairs'][1]['candidates']) == 0
    assert portal['coordinate_contract']['frame_id'] == 'bi_panorama_local_xz_v1'
    salve = {'schema_version': CANDIDATE_VERSION, 'method': 'salve_dav2',
             'coordinate_contract': copy.deepcopy(portal['coordinate_contract']), 'pairs': []}
    union = build_union(index, portal, salve)
    assert union['pair_denominator'] == 3
    assert union['union_candidate_denominator'] == 1
    assert union['methods']['salve_dav2']['missing_pairs'] == 3


def test_rejects_wrong_scope_frame_and_missing_pairs():
    data = prediction(); data['ground_truth_read'] = True
    with pytest.raises(ValueError, match='prediction-only'): export_prediction(data)
    data = prediction(); data['observations'][0]['prediction']['coordinate_spec']['units'] = 'meters'
    with pytest.raises(ValueError, match='coordinate'): export_prediction(data)
    data = prediction(); data['selected_pairs'].pop()
    with pytest.raises(ValueError, match='denominator'): export_prediction(data)


def test_invalid_hypothesis_still_counted():
    data = prediction()
    data['selected_pairs'][0]['pose_hypotheses'][0]['T_B_to_A'] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    _, portal = export_prediction(data)
    candidate = portal['pairs'][0]['candidates'][0]
    assert candidate['conversion_status'] == 'failed'
    assert candidate['T_B_to_A'] is None
