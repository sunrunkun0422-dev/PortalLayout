import hashlib
import json

import pytest

from tools.research10_frontend_cache import selected_manifests, rgb_hashes


def make_cohort(tmp_path):
    roles = {'fit': ['a'], 'selection': ['b'], 'calibration': ['c'], 'audit': ['d']}
    rows, hashes = [], {}
    for role, houses in roles.items():
        house = houses[0]
        path = tmp_path / (house+'.json')
        payload = {'floor_uid': house, 'observations': [dict(observation_id=house,
                    rgb={'relative_path': house+'.jpg'})]}
        path.write_text(json.dumps(payload))
        hashes[house] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(role=role, house_id=house, floor_uid=house, observations=1,
                         input_manifest=path.name))
        (tmp_path/(house+'.jpg')).write_bytes(house.encode())
    return {'roles': roles, 'cohort': rows, 'input_manifest_sha256': hashes}


def test_development_excludes_audit_and_detects_manifest_change(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.research10_frontend_cache.validate_public_manifest', lambda x: x)
    cohort = make_cohort(tmp_path)
    dev = selected_manifests(cohort, tmp_path, 'development')
    audit = selected_manifests(cohort, tmp_path, 'audit')
    assert {x['floor_uid'] for x in dev} == {'a', 'b', 'c'}
    assert [x['floor_uid'] for x in audit] == ['d']
    assert set(rgb_hashes(dev, tmp_path)) == {'a.jpg', 'b.jpg', 'c.jpg'}
    (tmp_path/'a.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest changed'):
        selected_manifests(cohort, tmp_path, 'development')


def test_role_overlap_and_path_traversal_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.research10_frontend_cache.validate_public_manifest', lambda x: x)
    cohort = make_cohort(tmp_path)
    cohort['roles']['audit'] = ['a']
    with pytest.raises(ValueError, match='roles overlap'):
        selected_manifests(cohort, tmp_path, 'development')
    cohort['roles']['audit'] = ['d']
    cohort['cohort'][0]['input_manifest'] = '../outside.json'
    with pytest.raises(ValueError):
        selected_manifests(cohort, tmp_path, 'development')
