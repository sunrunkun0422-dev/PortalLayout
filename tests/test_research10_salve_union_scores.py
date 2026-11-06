import copy

import numpy as np
import pytest
from PIL import Image

from tools.research10_candidate_comparison import build_union
from tools.research10_salve_union_scores import bind_scores
from tests.test_research10_candidate_comparison import inputs


def test_union_bound_scoring_retains_missing_and_checks_fingerprint(tmp_path):
    index, portal, salve = inputs()
    # Add a second distinct transform to the existing pair.
    second = copy.deepcopy(portal['pairs'][0]['candidates'][0])
    second['source_candidate_id'] = 'p3'
    second['T_B_to_A'][0][2] = 2.0
    portal['pairs'][0]['candidates'].append(second)
    union = build_union(index, portal, salve)
    ids = [c['candidate_id'] for c in union['candidates']]
    image = tmp_path / 'image.png'
    Image.fromarray(np.full((3, 4, 3), 127, dtype=np.uint8)).save(image)
    manifest = {'schema': 'salve-unlabeled-bev.v1', 'candidates': [
        {'candidate_id': ids[0], 'images': {k: 'image.png' for k in
            ('ceiling_a', 'ceiling_b', 'floor_a', 'floor_b')}}]}
    result, accounting, _ = bind_scores(union, manifest, lambda arrays: .75, base_dir=tmp_path)
    assert accounting['counts'] == {'scored': 1, 'unsupported': 1, 'failed': 0, 'not_reported': 0}
    assert result['union_fingerprint'] == union['union_fingerprint']
    assert next(x for x in result['scores'] if x['candidate_id'] == ids[0])['score'] == .75
    union['candidates'][0]['T_B_to_A'][0][2] += 1
    with pytest.raises(ValueError, match='fingerprint'):
        bind_scores(union, manifest, lambda _: .75, base_dir=tmp_path)


def test_reject_foreign_candidates_and_label_paths(tmp_path):
    union = build_union(*inputs())
    key = union['candidates'][0]['candidate_id']
    manifest = {'schema': 'salve-unlabeled-bev.v1', 'candidates': [
        {'candidate_id': 'foreign', 'images': {}}]}
    with pytest.raises(ValueError, match='outside'):
        bind_scores(union, manifest, lambda _: .75, base_dir=tmp_path)
    manifest['candidates'][0].update(candidate_id=key, images={'floor_a': 'incorrect_alignment/a.png'})
    with pytest.raises(ValueError, match='label-bearing'):
        bind_scores(union, manifest, lambda _: .75, base_dir=tmp_path)


def test_incomplete_and_failed_scores_remain_in_denominator(tmp_path):
    union = build_union(*inputs())
    key = union['candidates'][0]['candidate_id']
    manifest = {'schema': 'salve-unlabeled-bev.v1', 'candidates': [
        {'candidate_id': key, 'images': {}}]}
    _, accounting, _ = bind_scores(union, manifest, lambda _: .75, base_dir=tmp_path)
    assert accounting['candidate_denominator'] == 1
    assert accounting['counts']['unsupported'] == 1
