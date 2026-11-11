import pytest

from tools.research10_isolated_opening_manifests import select_records


def row(house, match=True):
    return {'id': 'pair_'+house, 'scene_id': house, 'image_a': house+'/a.jpg',
            'image_b': house+'/b.jpg', 'supervision': {'is_match': match}}


def test_house_disjoint_subsets_exclude_calibration_and_audit(tmp_path):
    roles = {'fit': ['01'], 'selection': ['02'], 'calibration': ['03'], 'audit': ['04']}
    for house in ('01', '02', '03', '04'):
        (tmp_path/house).mkdir()
        for name in ('a.jpg', 'b.jpg'): (tmp_path/house/name).write_bytes(b'x')
    result = select_records({'pairs': [row(h) for h in roles['fit']+roles['selection']+roles['audit']]}, roles, tmp_path)
    assert result['train'][0]['sourceHouseIds'] == ['01']
    assert result['val'][0]['sourceHouseIds'] == ['02']
    assert result['train'][0]['split'] == 'train' and result['val'][0]['split'] == 'val'
    assert len(result['train'][1]) == len(result['val'][1]) == 2


def test_cross_house_paths_and_role_overlap_fail_closed(tmp_path):
    roles = {'fit': ['01'], 'selection': ['02'], 'calibration': ['03'], 'audit': ['04']}
    bad = row('01'); bad['image_a'] = '04/a.jpg'
    with pytest.raises(ValueError, match='house identity'):
        select_records({'pairs': [bad, row('02')]}, roles, tmp_path)
    roles['audit'] = ['01']
    with pytest.raises(ValueError, match='roles overlap'):
        select_records({'pairs': []}, roles, tmp_path)
