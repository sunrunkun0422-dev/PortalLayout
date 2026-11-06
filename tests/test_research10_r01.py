import json

import pytest

from tools.research10_r01 import choose_cohort, recompute_geometry, sha256, verify_cache


def make_samples():
    groups = {'fit': [f'f{i:03}' for i in range(46)],
              'selection': [f's{i:03}' for i in range(22)],
              'calibration': [f'c{i:03}' for i in range(22)]}
    samples = []
    for houses in groups.values():
        for house in houses:
            for floor in range(2):
                samples.append(dict(house_id=house, floor_uid=f'{house}_{floor}',
                                    input_manifest=f'inputs/{house}_{floor}',
                                    observation_count=2, official_split='train'))
    samples.append(dict(house_id='test_only', floor_uid='test_floor', input_manifest='t',
                        observation_count=30, official_split='test'))
    return groups, samples


def test_cohort_is_deterministic_isolated_and_restricts_official_split():
    protocol, samples = make_samples()
    a, rows = choose_cohort(samples, protocol, {'f000', 's000', 'c000'})
    b, reverse = choose_cohort(list(reversed(samples)), protocol, {'f000', 's000', 'c000'})
    assert a == b
    assert {r['floor_uid'] for r in rows} == {r['floor_uid'] for r in reverse}
    assert {k: len(v) for k, v in a.items()} == dict(fit=40, selection=10, calibration=10, audit=10)
    all_houses = [h for houses in a.values() for h in houses]
    assert len(set(all_houses)) == 70
    assert not set(all_houses) & {'f000', 's000', 'c000', 'test_only'}
    assert set(a['selection']) <= set(protocol['selection'])
    assert set(a['calibration']) <= set(protocol['calibration'])


def test_cohort_insufficient_pool_fails_closed():
    protocol, samples = make_samples()
    with pytest.raises(ValueError, match='insufficient eligible houses'):
        choose_cohort(samples, protocol, protocol['fit'][:10])


def test_cache_content_hash_detects_corruption(tmp_path):
    folder = tmp_path / 'observations' / 'view1'
    folder.mkdir(parents=True)
    tensor = folder / 'key.npz'
    tensor.write_bytes(b'model data')
    (tmp_path / 'cache_contract.json').write_text(json.dumps({'fingerprint': 'fp'}))
    (tmp_path / 'run_report.json').write_text(json.dumps({'unique_observations': 1}))
    (folder / 'key.json').write_text(json.dumps(dict(contract_fingerprint='fp', observation_id='view1',
        tensor_file='observations/view1/key.npz', tensor_sha256=sha256(tensor))))
    assert verify_cache(tmp_path)['ok']
    tensor.write_bytes(b'changed')
    assert not verify_cache(tmp_path)['ok']


def test_paired_geometry_denominator_and_duplicate_check(tmp_path):
    p = tmp_path / 'geometry.csv'
    p.write_text('pair_id,house_id,collection,dominant_wall_s1p0,local_m2_s0p0\n'
                 'a,h1,r3,0,1\nb,h2,r3,1,1\n')
    assert recompute_geometry(p)['r3'] == dict(denominator=2, dominant_wall_s1p0_success=1,
                                             local_m2_s0p0_success=2, houses=2)
    with p.open('a') as f:
        f.write('a,h1,r3,0,1\n')
    with pytest.raises(ValueError, match='duplicate'):
        recompute_geometry(p)
