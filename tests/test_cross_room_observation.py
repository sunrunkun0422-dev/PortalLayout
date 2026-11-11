import math

import numpy as np
import pytest

from portallayout.geometry.conversion import xyz2uv
from portallayout.geometry.cross_room_observation import project_floor_boundary
from portallayout.geometry.cross_room_refinement import refine_local_layout


def project(uv, height=1., *, valid=None, confidence=None,
            units='camera_height_normalized', **kwargs):
    if valid is None:
        valid = [True] * len(uv)
    if confidence is None:
        confidence = [1.] * len(uv)
    return project_floor_boundary(uv, height, valid=valid, confidence=confidence,
                                  height_units=units, **kwargs)


def test_cardinal_rays_follow_bi_xz_and_height_scaling():
    uv = [[.5, .75], [.75, .75], [.25, .75], [0., .75], [1., .75]]
    expected = [[0., 1.5], [1.5, 0.], [-1.5, 0.], [0., -1.5], [0., -1.5]]
    result = project(uv, 1.5, units='meters')
    np.testing.assert_allclose(result['boundary_points'], expected, atol=1e-12)
    assert result['coordinate_spec']['frame'] == 'bi_panorama_local_xz'
    assert result['coordinate_spec']['units'] == 'meters'
    assert result['projection_status'] == 'projected'
    assert result['valid_runs'] == [[0, 1, 2, 3, 4]]


def test_seam_is_periodic_and_nearby_points_remain_nearby():
    result = project([[0., .75], [1., .75], [.000001, .75], [.999999, .75]])
    p = np.asarray(result['boundary_points'])
    np.testing.assert_allclose(p[0], p[1], atol=1e-15)
    assert np.linalg.norm(p[2] - p[3]) < 2e-5
    assert p[2, 0] < 0 < p[3, 0]


def test_projection_roundtrip_for_known_source_ground_points():
    xyz = np.array([[.2, 1., 2.], [-1.2, 1., .3], [.5, 1., -3.], [-.3, 1., -2.]])
    uv = xyz2uv(xyz)
    result = project(uv)
    np.testing.assert_allclose(result['boundary_points'], xyz[:, [0, 2]], atol=1e-12)
    assert result['source_indices'] == [0, 1, 2, 3]
    assert not {'status', 'visibility', 'pose_status'} & result.keys()


def test_invalid_horizon_and_upper_hemisphere_are_not_projected():
    # Near-horizontal ray is within the declared vertical component margin.
    uv = [[.5, .25], [.5, .5], [.5, .5 + 1e-8], [.5, .75], [.5, 1.], [.5, 1.1]]
    result = project(uv)
    assert result['source_indices'] == [3, 4]
    assert result['point_reasons'] == ['at_or_above_horizon'] * 3 + [None, None, 'uv_out_of_range']
    assert result['projection_status'] == 'partially_projected'
    np.testing.assert_allclose(result['boundary_points'][1], [0, 0], atol=1e-15)


def test_caller_mask_bad_uv_and_confidences_preserve_run_boundaries():
    uv = [[.5, .75], [.5, .75], [float('nan'), .75], [.5, .75], [.5, .75], [.5, .75]]
    result = project(uv, valid=[True, False, True, True, True, True],
                     confidence=[.9, .1, .8, 1.1, .4, .3])
    assert result['source_indices'] == [0, 4, 5]
    assert result['point_confidences'] == [.9, .4, .3]
    assert result['point_reasons'] == [None, 'caller_invalid', 'nonfinite_uv', 'invalid_confidence', None, None]
    assert result['valid_runs'] == [[0], [1, 2]]
    assert math.isnan(uv[2][0])  # caller's input is not mutated


def test_missing_or_bad_height_never_falls_back_to_fake_scale():
    for height in [None, 0., -1., float('nan'), float('inf'), True]:
        result = project([[.5, .75]], height=height, units='meters')
        assert result['projection_status'] == 'no_valid_points'
        assert result['boundary_points'] == []
        assert result['point_reasons'] == ['camera_height_unavailable']
    with pytest.raises(ValueError, match='requires camera_height=1'):
        project([[.5, .75]], 1.5)


def test_explicit_contract_and_empty_input():
    with pytest.raises(ValueError, match='boolean'):
        project([[.5, .75]], valid=[1])
    with pytest.raises(ValueError, match='confidence'):
        project_floor_boundary([[.5, .75]], 1., valid=[True], confidence=[], height_units='meters')
    with pytest.raises(ValueError, match='height_units'):
        project([[.5, .75]], units='guessed')
    with pytest.raises(ValueError, match='horizon_epsilon'):
        project([[.5, .75]], horizon_epsilon=0.)
    result = project([], valid=[], confidence=[])
    assert result['projection_status'] == 'no_valid_points'
    assert result['boundary_points'] == result['source_indices'] == result['point_reasons'] == []


def test_refinement_requires_separate_verified_metadata_and_transform():
    # Synthetic points lie 0.1 camera-height units beyond the predicted wall.
    local = np.array([[-.5, 1., 1.1], [.5, 1., 1.1]])
    projected = project(xyz2uv(local))
    assert 'status' not in projected and 'pose_status' not in projected
    polygon = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
    metadata = {'source_observation_id': 'source', 'target_observation_id': 'target',
                'source_opening_id': 'open_a', 'target_opening_id': 'open_b',
                'frame': 'source', 'boundary_points': projected['boundary_points'],
                'confidence': .95}
    rejected = refine_local_layout(polygon, [metadata], {'source': np.eye(3).tolist()})
    assert rejected['status'] == 'unchanged'
    assert rejected['reason'] == 'no_usable_observation'
    verified = dict(metadata, status='accepted', visibility='visible', pose_status='accepted')
    result = refine_local_layout(polygon, [verified], {'source': np.eye(3).tolist()})
    assert result['status'] == 'refined'
    assert result['changed_vertex_indices'] == [2, 3]
