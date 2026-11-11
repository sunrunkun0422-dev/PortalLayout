import json
from types import SimpleNamespace

import pytest

from portallayout.datasets.floor_auxiliary_supervision import (
    house_protocol, house_role, validate_cache_house_roles,
    validate_supervision_house_roles, validate_auxiliary_for_training,
)
from portallayout.engine.opening_pair_training import read_supervision
from portallayout.utils.room_feature_cache import fingerprint, content_hash
from tools.report_opening_matches import frozen_audit_threshold


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def setup_protocol(tmp_path):
    root = tmp_path/'dataset'
    roles = dict(fit=['0001'], selection=['0002'], calibration=['0003'], audit=['0004'])
    write_json(root/'splits/official_house_split.json',
               dict(train=['0001', '0002', '0003', '0004', '0005'], val=['0006'], test=['0007']))
    write_json(root/'splits/derived_house_split.json', dict(fit=['0001'], dev=['0002', '0003']))
    rows = [dict(sample_id='s'+house, floor_uid='f'+house, house_id=house, official_split='train',
                 input_manifest='inputs/'+house+'.json')
            for house in ('0001', '0002', '0003', '0004', '0005')]
    (root/'samples').mkdir()
    (root/'samples/all_panoramas.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for row in rows:
        obs = 'v'+row['house_id']
        write_json(root/row['input_manifest'], dict(schema_version='zind-floorset.input.v1',
            sample_id=row['sample_id'], floor_uid=row['floor_uid'], task='unposed_floor_layout_from_panoramas',
            input_mode='panorama_only', selection_policy='all_panoramas', projection='equirectangular',
            pose_prior='none', calibration_mode='scale_unknown', observation_order=[obs],
            observations=[dict(observation_id=obs, rgb=dict(root_key='zind_raw', relative_path='img.jpg'),
                               image_size_wh=[100, 50])]))
    files = tmp_path/'protocol'
    write_json(files/'cohort.json', {'roles': roles,
        'cohort': [dict(house_id=house, floor_uid='f'+house, role=role)
                   for role, houses in roles.items() for house in houses]})
    protocol = {'schema_version': 'research10.house-protocol.v1', 'seed': 123, **roles,
                'source_sha256': content_hash(files/'cohort.json')}
    protocol['fingerprint'] = fingerprint(protocol)
    write_json(files/'house_protocol.json', protocol)
    return root, files, protocol


def test_custom_protocol_and_legacy_default(tmp_path):
    root, files, expected = setup_protocol(tmp_path)
    assert house_protocol(root, 123, files/'house_protocol.json') == expected
    assert house_role('0004', expected) == 'audit'
    assert house_role('0005', expected) == 'heldout'
    legacy = house_protocol(root)
    assert 'audit' not in legacy and house_role('0004', legacy) == 'heldout'


def test_modified_protocol_and_cohort_rejected(tmp_path):
    root, files, protocol = setup_protocol(tmp_path)
    protocol['fit'].append('0005')
    write_json(files/'house_protocol.json', protocol)
    with pytest.raises(ValueError, match='fingerprint'):
        house_protocol(root, protocol_path=files/'house_protocol.json')
    root, files, protocol = setup_protocol(tmp_path/'other')
    write_json(files/'cohort.json', {'roles': {}})
    with pytest.raises(ValueError, match='cohort source'):
        house_protocol(root, protocol_path=files/'house_protocol.json')


@pytest.mark.parametrize('roles, message', [
    ({'fit': ['0001', '0001']}, 'duplicate'),
    ({'audit': ['0001']}, 'overlap'),
    ({'audit': ['0007']}, 'outside official train'),
])
def test_duplicate_overlap_and_test_house_rejected(tmp_path, roles, message):
    root, files, protocol = setup_protocol(tmp_path)
    protocol.update(roles)
    protocol.pop('fingerprint')
    protocol['fingerprint'] = fingerprint(protocol)
    write_json(files/'house_protocol.json', protocol)
    with pytest.raises(ValueError, match=message):
        house_protocol(root, protocol_path=files/'house_protocol.json')


def test_cache_and_supervision_reject_audit_heldout_and_spoofing(tmp_path):
    root, files, protocol = setup_protocol(tmp_path)
    allowed = {'fit', 'selection', 'calibration'}
    def cache(house):
        return SimpleNamespace(floors=[dict(sample_id='s'+house, floor_uid='f'+house,
                                            observation_order=['v'+house])])
    assert validate_cache_house_roles(cache('0001'), root, protocol, allowed) == {'f0001': '0001'}
    for house, reason in [('0004', 'audit'), ('0005', 'heldout')]:
        with pytest.raises(ValueError, match=reason):
            validate_cache_house_roles(cache(house), root, protocol, allowed)
        with pytest.raises(ValueError, match=reason):
            validate_supervision_house_roles({'pairs': [{'house_id': house}]}, protocol, allowed)
    with pytest.raises(ValueError, match='floor/house'):
        validate_supervision_house_roles({'pairs': [{'house_id': '0001', 'floor_uid': 'f0004'}]},
                                         protocol, allowed, {'f0004': '0004'})
    with pytest.raises(ValueError, match='forbidden audit'):
        validate_auxiliary_for_training(write_aux(files, protocol), protocol)


def write_aux(files, protocol):
    path = files/'aux.json'
    write_json(path, {'house_protocol': protocol, 'records': [{'house_id': '0004', 'role': 'audit'}]})
    return path


def test_audit_only_labels_rejected_by_training_loader(tmp_path):
    root, files, protocol = setup_protocol(tmp_path)
    cache = SimpleNamespace(snapshot='same', records={})
    data = dict(source_snapshot='same', usage='audit_only', house_protocol=protocol, pairs=[])
    data['fingerprint'] = fingerprint(data)
    p = files/'weak.json'; write_json(p, data)
    with pytest.raises(ValueError, match='audit-only'):
        read_supervision(p, cache)
    assert read_supervision(p, cache, allow_audit=True)['usage'] == 'audit_only'


def test_audit_uses_only_frozen_calibration_and_exact_model(tmp_path):
    _, files, protocol = setup_protocol(tmp_path)
    checkpoint = files/'checkpoint'; checkpoint.write_bytes(b'weights')
    report = dict(house_protocol=protocol, pair_checkpoint_sha256=content_hash(checkpoint),
                  calibration={'status': 'calibrated', 'threshold': .8}, roles={'calibration': {}})
    report['fingerprint'] = fingerprint(report)
    path = files/'calibration.json'; write_json(path, report)
    assert frozen_audit_threshold(path, protocol, checkpoint)[1] == .8
    checkpoint.write_bytes(b'other')
    with pytest.raises(ValueError, match='does not match'):
        frozen_audit_threshold(path, protocol, checkpoint)
    checkpoint.write_bytes(b'weights')
    report.pop('fingerprint'); report['roles']['audit'] = {}; report['fingerprint'] = fingerprint(report)
    write_json(path, report)
    with pytest.raises(ValueError, match='does not match'):
        frozen_audit_threshold(path, protocol, checkpoint)


@pytest.mark.parametrize('command', ['prepare', 'train'])
def test_cli_rejects_audit_cache_before_supervision_or_training(tmp_path, monkeypatch, command):
    from tools.run_floor_layout import main
    import portallayout.utils.frozen_opening_cache as frozen
    root, files, _ = setup_protocol(tmp_path)
    cache = SimpleNamespace(floors=[dict(sample_id='s0004', floor_uid='f0004', observation_order=['v0004'])])
    monkeypatch.setattr(frozen, 'FrozenOpeningCache', lambda _: cache)
    args = [command, '--dataset', str(root), '--prediction-cache', 'unused',
            '--house-protocol', str(files/'house_protocol.json'), '--output', str(tmp_path/'out'), '--device', 'cpu']
    if command == 'train': args += ['--stage', 'pairs', '--mode', 'smoke']
    with pytest.raises(ValueError, match='forbidden audit'):
        main(args)
