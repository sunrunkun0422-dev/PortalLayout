import copy

import numpy as np
import pytest

from tools.research10_extended_proposals import propose


def record(interval=(6, 1)):
    return {'observation_id': 'room_a', 'prediction': {'coordinate_spec': {
        'frame': 'bi_panorama_local_xz', 'units': 'camera_height_normalized'},
        'openings': [{'candidate_id': 'portal1', 'token_interval_inclusive': list(interval)}]}}


def test_cross_seam_preserves_circular_order_and_never_confirms_target():
    arrays = {'depth_enclosed': np.ones(8), 'depth_extended': np.array([2, 2, 1, 1, 1, 1, 2, 2])}
    original = copy.deepcopy(record())
    result = propose(original, arrays)
    row = result['openings'][0]
    assert row['runs'][0]['token_indices'] == [6, 7, 0, 1]
    assert len(row['runs'][0]['boundary_points']) == 4
    assert row['visibility'] == row['target_membership'] == 'unknown'
    assert result['accepted_for_refinement'] is False
    assert original == record()


def test_invalid_or_nonextended_depth_breaks_runs():
    arrays = {'depth_enclosed': np.ones(8), 'depth_extended': np.array([2, 2, 2, 0, 2, 2, 2, 2.])}
    result = propose(record((0, 5)), arrays)
    assert [r['token_indices'] for r in result['openings'][0]['runs']] == [[0, 1, 2], [4, 5]]
    arrays['depth_extended'][:] = 1.
    assert propose(record((0, 5)), arrays)['openings'][0]['status'] == 'no_candidate'


def test_wrong_coordinate_or_opening_identity_is_rejected():
    arrays = {'depth_enclosed': np.ones(8), 'depth_extended': np.ones(8)*2}
    bad = record(); bad['prediction']['coordinate_spec']['units'] = 'meters'
    with pytest.raises(ValueError): propose(bad, arrays)
    bad = record(); bad['prediction']['openings'].append(copy.deepcopy(bad['prediction']['openings'][0]))
    with pytest.raises(ValueError): propose(bad, arrays)
