import json
import math

import numpy as np
import pytest

from tools.research10_candidate_comparison import (
    PAIR_VERSION, CANDIDATE_VERSION, SCORE_VERSION, build_union, score_accounting, main,
)


def inputs():
    index = {'schema_version': PAIR_VERSION, 'pairs': [
        dict(floor_uid='floor1', observation_a='A', observation_b='B'),
        dict(floor_uid='floor1', observation_a='A', observation_b='C')]}
    contract = dict(frame_id='common_xy_v1', units='height_unit',
                    transform_direction='B_to_A', equation='p_A=T_B_to_A*p_B')
    def candidate(name, transform, a='a1', b='b1', status='converted', reason=None):
        return dict(source_candidate_id=name, source_opening_a=a, source_opening_b=b,
                    conversion_status=status, reason=reason, T_B_to_A=transform)
    t = [[1., 0., 1.], [0., 1., 0.], [0., 0., 1.]]
    portal = {'schema_version': CANDIDATE_VERSION, 'method': 'portallayout',
              'coordinate_contract': contract.copy(), 'pairs': [
        dict(floor_uid='floor1', observation_a='A', observation_b='B',
             candidates=[candidate('p1', t), candidate('p2', None, status='unsupported', reason='no_conversion')]),
        dict(floor_uid='floor1', observation_a='A', observation_b='C', candidates=[])]}
    salve = {'schema_version': CANDIDATE_VERSION, 'method': 'salve_dav2',
             'coordinate_contract': contract.copy(), 'pairs': [
        dict(floor_uid='floor1', observation_a='B', observation_b='A',
             candidates=[candidate('s1', [[1., 0., -1.], [0., 1., 0.], [0., 0., 1.]], 'b2', 'a2')])]}
    return index, portal, salve


def test_fixed_denominator_dedup_reverse_and_no_label_leak():
    index, portal, salve = inputs()
    result = build_union(index, portal, salve)
    assert result['pair_denominator'] == 2
    assert result['union_candidate_denominator'] == 1
    assert result['methods']['salve_dav2']['missing_pairs'] == 1
    assert result['methods']['portallayout']['failed_or_unsupported_conversions'] == 1
    c = result['candidates'][0]
    np.testing.assert_allclose(c['T_B_to_A'], [[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    assert len(c['sources']) == 2
    salve_ref = next(s for s in c['sources'] if s['method'] == 'salve_dav2')
    assert (salve_ref['source_opening_a'], salve_ref['source_opening_b']) == ('a2', 'b2')
    assert len(result['conversion_failures']) == 1
    assert 'candidate_label' not in json.dumps(result)


def test_input_order_does_not_change_union_identity():
    index, portal, salve = inputs()
    first = build_union(index, portal, salve)
    index['pairs'].reverse(); portal['pairs'].reverse(); portal['pairs'][1]['candidates'].reverse()
    second = build_union(index, portal, salve)
    assert first['union_fingerprint'] == second['union_fingerprint']
    assert first['candidates'] == second['candidates']


def test_gt_fields_and_source_identifiers_rejected():
    index, portal, salve = inputs()
    portal['pairs'][0]['candidates'][0]['gt_label'] = 1
    with pytest.raises(ValueError, match='exactly'):
        build_union(index, portal, salve)
    index, portal, salve = inputs()
    portal['pairs'][0]['candidates'][0]['source_candidate_id'] = 'gt_alignment_approx'
    with pytest.raises(ValueError, match='label-bearing'):
        build_union(index, portal, salve)


def test_coordinate_mismatch_and_outside_fixed_pair_rejected():
    index, portal, salve = inputs()
    salve['coordinate_contract']['units'] = 'meters'
    with pytest.raises(ValueError, match='coordinate contracts differ'):
        build_union(index, portal, salve)
    index, portal, salve = inputs()
    salve['pairs'][0]['observation_a'] = 'unknown'
    with pytest.raises(ValueError, match='absent from frozen index'):
        build_union(index, portal, salve)


@pytest.mark.parametrize('bad_matrix', [
    [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[1., 0., 0.], [0., 2., 0.], [0., 0., 1.]],
    [[1., 0., float('nan')], [0., 1., 0.], [0., 0., 1.]],
    [[1., 0.], [0., 1.]],
])
def test_invalid_converted_pose_counted_not_dropped(bad_matrix):
    index, portal, salve = inputs()
    portal['pairs'][0]['candidates'][0]['T_B_to_A'] = bad_matrix
    result = build_union(index, portal, salve)
    assert result['methods']['portallayout']['input_candidates'] == 2
    assert result['methods']['portallayout']['failed_or_unsupported_conversions'] == 2
    assert result['union_candidate_denominator'] == 1


def test_scoring_coverage_and_binding():
    result = build_union(*inputs())
    key = result['candidates'][0]['candidate_id']
    empty = score_accounting(result, 'portallayout')
    assert empty['counts']['not_reported'] == 1
    scores = dict(schema_version=SCORE_VERSION, method='portallayout',
                  union_fingerprint=result['union_fingerprint'],
                  scores=[dict(candidate_id=key, status='unsupported', score=None, reason='no_seed_mapping')])
    accounting = score_accounting(result, 'portallayout', scores)
    assert accounting['candidate_denominator'] == 1
    assert accounting['counts']['unsupported'] == 1
    scores['union_fingerprint'] = 'wrong'
    with pytest.raises(ValueError, match='schema or method'):
        score_accounting(result, 'portallayout', scores)
    scores['union_fingerprint'] = result['union_fingerprint']
    scores['scores'][0].update(status='scored', score=math.nan, reason=None)
    with pytest.raises(ValueError, match='finite score'):
        score_accounting(result, 'portallayout', scores)


def test_cli_outputs_identical_candidate_inputs_without_scores(tmp_path, monkeypatch):
    index, portal, salve = inputs()
    paths = []
    for name, item in [('index.json', index), ('portal.json', portal), ('salve.json', salve)]:
        path = tmp_path/name; path.write_text(json.dumps(item)); paths.append(path)
    output = tmp_path/'out'
    monkeypatch.setattr('sys.argv', ['candidate-comparison', '--pairs', str(paths[0]),
        '--portal-candidates', str(paths[1]), '--salve-candidates', str(paths[2]), '--output', str(output)])
    main()
    assert (output/'portallayout_scoring_input.jsonl').read_text() == (output/'salve_dav2_scoring_input.jsonl').read_text()
    status = json.loads((output/'scoring_accounting.json').read_text())
    assert status['portallayout']['counts']['not_reported'] == 1
    assert status['salve_dav2']['counts']['not_reported'] == 1
    with pytest.raises(ValueError, match='new or empty'):
        main()


def test_cli_rejects_label_based_input_paths(tmp_path, monkeypatch):
    index, portal, salve = inputs()
    folder = tmp_path/'gt_alignment_approx'; folder.mkdir()
    paths = []
    for name, item in [('index.json', index), ('portal.json', portal), ('salve.json', salve)]:
        path = folder/name; path.write_text(json.dumps(item)); paths.append(path)
    monkeypatch.setattr('sys.argv', ['candidate-comparison', '--pairs', str(paths[0]),
        '--portal-candidates', str(paths[1]), '--salve-candidates', str(paths[2]), '--output', str(tmp_path/'out')])
    with pytest.raises(ValueError, match='label-bearing input paths'):
        main()
    assert not (tmp_path/'out').exists()
