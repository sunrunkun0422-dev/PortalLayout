import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.research10_salve_scoring import ROLES, score_manifest, validate_manifest


def write_images(directory):
    directory.mkdir(parents=True)
    result = {}
    for i, role in enumerate(ROLES):
        name = role + '.png'
        Image.new('RGB', (5, 4), (10 + i, 20 + 2 * i, 30 + i)).save(directory / name)
        result[role] = str(directory / name)
    return result


def manifest(*rows):
    return {'schema': 'salve-unlabeled-bev.v1', 'candidates': list(rows)}


def mock_scorer(images):
    assert len(images) == 4
    assert all(isinstance(x, np.ndarray) and x.shape == (4, 5, 3) for x in images)
    # The expected role order is encoded by the synthetic pixel colors.
    assert [int(x[0, 0, 0]) for x in images] == [10, 11, 12, 13]
    return float(sum(int(x.sum()) for x in images) / 100000.)


def test_directory_label_flip_does_not_affect_input_or_output(tmp_path):
    first = write_images(tmp_path / 'gt_alignment_approx')
    other_dir = tmp_path / 'incorrect_alignment'
    shutil.copytree(tmp_path / 'gt_alignment_approx', other_dir)
    second = {role: str(other_dir / (role + '.png')) for role in ROLES}
    left = score_manifest(manifest({'candidate_id': 'c1', 'images': first}), mock_scorer, base_dir=tmp_path)
    right = score_manifest(manifest({'candidate_id': 'c1', 'images': second}), mock_scorer, base_dir=tmp_path)
    assert left == right
    encoded = json.dumps(left)
    assert 'gt_alignment' not in encoded and 'incorrect_alignment' not in encoded
    assert not ({'y_true', 'label', 'target'} & set(left['rows'][0]))
    assert left['scored_count'] == 1


def test_missing_and_unreadable_are_counted_per_candidate(tmp_path):
    images = write_images(tmp_path / 'inputs')
    Path(images['floor_a']).unlink()
    Path(images['floor_b']).write_text('not an image')
    one = {'candidate_id': 'missing', 'images': images}
    two = {'candidate_id': 'no_role', 'images': {}}
    result = score_manifest(manifest(one, two), lambda images: pytest.fail('model must not run'), base_dir=tmp_path)
    assert result['candidate_count'] == result['incomplete_count'] == 2
    assert result['scored_count'] == 0
    assert result['rows'][0]['missing_roles'] == ['floor_a']
    assert result['rows'][0]['unreadable_roles'] == ['floor_b']
    assert result['rows'][1]['missing_roles'] == list(ROLES)
    assert [x['candidate_id'] for x in result['rows']] == ['missing', 'no_role']
    assert str(tmp_path) not in json.dumps(result)


def test_duplicate_id_and_labels_are_rejected_before_model(tmp_path):
    row = {'candidate_id': 'c1', 'images': {}}
    with pytest.raises(ValueError, match='duplicate'):
        score_manifest(manifest(row, row), lambda images: pytest.fail('not called'), base_dir=tmp_path)
    with pytest.raises(ValueError, match='only'):
        validate_manifest(manifest(dict(row, label=1)))
    with pytest.raises(ValueError, match='only'):
        validate_manifest(dict(manifest(row), y_true=1))
    with pytest.raises(ValueError, match='four BEV roles'):
        validate_manifest(manifest({'candidate_id': 'c2', 'images': {'label': 'fake'}}))


def test_model_failure_and_bad_probability_stay_in_denominator(tmp_path):
    images = write_images(tmp_path / 'inputs')
    rows = manifest({'candidate_id': 'bad', 'images': images}, {'candidate_id': 'okay', 'images': images})
    calls = iter([float('nan'), .75])
    result = score_manifest(rows, lambda imgs: next(calls), base_dir=tmp_path)
    assert result['scoring_failed_count'] == 1 and result['scored_count'] == 1
    assert result['rows'][0] == {'candidate_id': 'bad', 'positive_probability': None,
                                 'status': 'scoring_failed', 'reason': 'model_or_probability_error'}
    assert result['rows'][1]['positive_probability'] == .75


def test_relative_paths_are_resolved_only_as_image_sources(tmp_path):
    write_images(tmp_path / 'inputs')
    relative = {role: 'inputs/' + role + '.png' for role in ROLES}
    result = score_manifest(manifest({'candidate_id': 'relative', 'images': relative}), mock_scorer, base_dir=tmp_path)
    assert result['rows'][0]['status'] == 'scored'
    assert 'inputs/' not in json.dumps(result)
