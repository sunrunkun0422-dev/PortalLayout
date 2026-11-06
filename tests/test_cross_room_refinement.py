import copy

import numpy as np
import pytest
from shapely.geometry import Polygon

from portallayout.geometry.cross_room_refinement import RefinementConfig, refine_local_layout
from portallayout.geometry.floor_geometry import matrix, points_transform


def observation(points, **updates):
    row = dict(source_observation_id='B', target_observation_id='A',
               source_opening_id='door-b', target_opening_id='door-a',
               frame='target', boundary_points=points, confidence=.95,
               visibility='visible', status='accepted', pose_status='accepted')
    row.update(updates)
    return row


def test_only_supported_wall_moves_towards_observation_and_input_is_unchanged():
    initial = [[0., 0.], [4., 0.], [4., 3.], [0., 3.]]
    rows = [observation([[1., .2], [2., .2], [3., .2]])]
    before = copy.deepcopy((initial, rows))
    result = refine_local_layout(initial, rows, {})
    assert (initial, rows) == before
    assert result['status'] == 'refined'
    assert set(result['changed_vertex_indices']) == {0, 1}
    assert 0 < result['polygon'][0][1] < .2
    assert result['polygon'][2:] == initial[2:]
    assert Polygon(result['polygon']).is_valid
    assert result['max_displacement'] <= RefinementConfig().max_displacement


def test_source_transform_matches_equivalent_target_points():
    initial = [[0., 0.], [4., 0.], [4., 3.], [0., 3.], [0., 0.]]
    transform = matrix([1., -.3, .4, 0.])
    target = np.array([[1., .2], [2., .2], [3., .2]])
    source = points_transform(target, np.linalg.inv(transform))
    a = refine_local_layout(initial, [observation(target.tolist())], {})
    b = refine_local_layout(initial, [observation(source.tolist(), frame='source')], {'B': transform})
    assert a['status'] == b['status'] == 'refined'
    np.testing.assert_allclose(a['polygon'], b['polygon'], atol=1e-8)
    assert b['polygon'][0] == b['polygon'][-1]


@pytest.mark.parametrize('changes, expected', [
    ({'visibility': 'unknown'}, 'unverified_or_not_visible'),
    ({'status': 'rejected'}, 'unverified_or_not_visible'),
    ({'pose_status': 'unknown'}, 'unverified_or_not_visible'),
    ({'confidence': float('nan')}, 'invalid_or_low_confidence'),
    ({'boundary_points': [[9., 9.], [10., 9.]]}, 'no_compatible_wall'),
    ({'boundary_points': [[1., 1.], [1., 1.]]}, 'degenerate_boundary'),
    ({'frame': 'source'}, 'invalid_observation_or_transform'),
])
def test_rejected_or_unusable_evidence_leaves_input_unchanged(changes, expected):
    initial = [[0., 0.], [4., 0.], [4., 3.], [0., 3.]]
    result = refine_local_layout(initial, [observation([[1., .2], [2., .2]], **changes)], {})
    assert result['polygon'] == initial
    assert result['status'] == 'unchanged'
    assert result['observations'][0]['reason'] == expected


def test_non_manhattan_polygon_and_wrong_orientation():
    initial = [[0., 0.], [4., 1.], [3., 4.], [-.5, 2.]]
    good = observation([[1., .45], [2., .7], [3., .95]])
    result = refine_local_layout(initial, [good], {})
    assert result['status'] == 'refined'
    assert Polygon(result['polygon']).is_valid
    bad = observation([[2., .1], [2., .4]])
    assert refine_local_layout(initial, [bad], {})['status'] == 'unchanged'


def test_invalid_input_and_transform_are_not_silently_repaired():
    with pytest.raises(ValueError):
        refine_local_layout([[0, 0], [1, 1], [1, 0], [0, 1]], [], {})
    initial = [[0, 0], [4, 0], [4, 3], [0, 3]]
    bad = observation([[1, 0], [2, 0]], frame='source')
    assert refine_local_layout(initial, [bad], {'B': np.diag([-1, 1, 1])})['status'] == 'unchanged'


def test_observations_for_different_targets_cannot_be_mixed():
    initial = [[0, 0], [4, 0], [4, 3], [0, 3]]
    rows = [observation([[1, .2], [2, .2]]),
            observation([[2, .2], [3, .2]], target_observation_id='different')]
    result = refine_local_layout(initial, rows, {})
    assert result['observations'][1]['reason'] == 'different_target_observation'
