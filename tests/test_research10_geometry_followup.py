import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import research10_geometry_followup as follow


def signed(path, value, field):
    value = dict(value)
    value[field] = hashlib.sha256(follow.canonical(value)).hexdigest()
    follow.write_json_atomic(path, value)
    return value


def test_unknown_hypotheses_are_not_called_wrong():
    assert follow.classify_labels([0, -1]) == 'unknown'
    assert follow.classify_labels([0, -1, 1]) == 'solvable'
    assert follow.classify_labels([0, 0]) == 'all_incorrect'
    assert follow.classify_labels([]) == 'no_hypotheses'
    assert follow.classify_labels([0], gt_valid=False) == 'gt_unavailable'
    with pytest.raises(ValueError, match='invalid'):
        follow.classify_labels([2])


def test_wait_rejects_failed_and_modified_pipeline(tmp_path):
    contract = signed(tmp_path/'contract.json', {'schema': 'test'}, 'fingerprint')
    follow.write_json_atomic(tmp_path/'status.json', {'status': 'blocked', 'contract_sha256': contract['fingerprint']})
    with pytest.raises(ValueError, match='did not complete'):
        follow.wait_completed(tmp_path, .01)
    follow.write_json_atomic(tmp_path/'contract.json', {'schema': 'tampered', 'fingerprint': contract['fingerprint']})
    with pytest.raises(ValueError, match='fingerprint'):
        follow.wait_completed(tmp_path, .01)


def test_diagnostic_job_requires_success_and_complete_checkpoint(tmp_path):
    job, front, target = tmp_path/'job', tmp_path/'front', tmp_path/'train'/'diagnostic_pairs'
    (job/'diagnostic_job').mkdir(parents=True); front.mkdir(); target.mkdir(parents=True)
    diagnostic = signed(job/'diagnostic_job/contract.json', dict(kind='floor-train', purpose='diagnostic',
        stage='pairs', output_dir=str(target)), 'contract_sha256')
    follow.write_json_atomic(job/'diagnostic_job/status.json', dict(status='completed', exit_code=0,
                                                                  contract_sha256=diagnostic['contract_sha256']))
    frontend = signed(front/'contract.json', dict(output_dir=str(tmp_path/'cache'), cache_scope='development'), 'contract_sha256')
    follow.write_json_atomic(front/'status.json', dict(status='completed', contract_sha256=frontend['contract_sha256']))
    report_path = target/'training_report.json'
    report = dict(training_status='trained', parameters_changed=True, steps=100, selected_step=50,
                  skipped_batches=0, gradient_norms_first_step={'encoder': 1., 'matcher': 2.})
    follow.write_json_atomic(report_path, report)
    (target/'checkpoint.pt').write_bytes(b'model')
    pipeline = {'output_dir': str(target.parent), 'frontend_job': str(front),
                'frontend_contract_sha256': frontend['contract_sha256']}
    state = {'diagnostic_report_sha256': follow.content_hash(report_path)}
    assert follow.diagnostic_inputs(job, pipeline, state)['checkpoint'] == (target/'checkpoint.pt').resolve()
    report['selected_step'] = None; follow.write_json_atomic(report_path, report)
    state['diagnostic_report_sha256'] = follow.content_hash(report_path)
    with pytest.raises(ValueError, match='valid training report'):
        follow.diagnostic_inputs(job, pipeline, state)


def test_summary_preserves_grey_zone_and_failed_pair_denominators(monkeypatch):
    import portallayout.evaluation.pose_geometry as geometry
    monkeypatch.setattr(geometry, 'select_geometry', lambda parts: {'status': 'diagnostic'})
    rows = [dict(floor_uid='f1'), dict(floor_uid='f2')]
    predictions = [dict(floor_uid='f1', status='predicted', pair_denominator=2),
                   dict(floor_uid='f2', status='failed', pair_denominator=3)]
    variant = {'v': {'labels': [0, -1]}}
    part = {'oracles': [{'association_or_annotation_status': 'unknown'}],
            'inventory': [dict(relationship='cross_room', gt_valid=True)],
            'rows': [dict(role='selection', collection='r3', relationship='cross_room',
                          gt_valid=True, variants=variant)]}
    audits = [dict(floor_uid='f1', status='audited', geometry=part,
                   selected_pair_outcomes=[{'outcome': 'unknown'}, {'outcome': 'incorrect'}]),
              dict(floor_uid='f2', status='failed')]
    result = follow.summarize_followup(rows, predictions, audits)
    assert result['pair_denominator'] == 5 and result['prediction_failed_pair_denominator'] == 3
    assert result['audit_failed_pair_denominator'] == 3
    assert result['variant_outcomes']['selection/r3/cross_room/v'] == {'unknown': 1}
    assert result['selected_pair_outcomes'] == {'unknown': 1, 'incorrect': 1}


def test_annotation_hashes_do_not_open_protected_roles(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(follow, 'content_hash', lambda p: opened.append(str(p)) or 'hash')
    for uid in ('fit_floor', 'sel_floor'):
        for filename in ('floor_gt.json', 'observations_gt.jsonl', 'portal_links.jsonl'):
            path = tmp_path/'labels'/uid/filename; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
    rows = [dict(role='fit', floor_uid='fit_floor'), dict(role='selection', floor_uid='sel_floor')]
    assert len(follow.annotation_hashes(tmp_path, rows)) == 6
    assert all('fit_floor' in p or 'sel_floor' in p for p in opened)
    with pytest.raises(ValueError, match='forbidden annotation role'):
        follow.annotation_hashes(tmp_path, [dict(role='calibration', floor_uid='cal_floor')])
    assert not any('cal_floor' in p for p in opened)


def test_cohort_rejects_duplicate_or_missing_house():
    protocol = {'fit': ['h1'], 'selection': ['h2'], 'calibration': ['h3'], 'audit': ['h4']}
    cohort = {'roles': protocol, 'cohort': [dict(role=role, house_id=houses[0], floor_uid='f'+houses[0])
                                           for role, houses in protocol.items()]}
    assert [r['role'] for r in follow.cohort_scope(cohort, protocol)] == ['fit', 'selection']
    cohort['cohort'][1]['house_id'] = 'h1'
    with pytest.raises(ValueError, match='duplicate cohort house'):
        follow.cohort_scope(cohort, protocol)


def test_resume_finishes_all_predictions_before_annotation(tmp_path, monkeypatch):
    import portallayout.datasets.floor_auxiliary_supervision as aux
    import portallayout.utils.frozen_opening_cache as frozen
    import portallayout.engine.floor_layout_pipeline as pipeline
    import portallayout.evaluation.pose_geometry as geometry

    job, front, dataset, train, cache_path = (tmp_path/p for p in ('job', 'front', 'dataset', 'train', 'cache'))
    for p in (job, front, dataset, train, cache_path): p.mkdir()
    protocol = {'fit': ['h1'], 'selection': ['h2'], 'calibration': ['h3'], 'audit': ['h4']}
    rows = [dict(role=role, house_id=houses[0], floor_uid='f'+houses[0]) for role, houses in protocol.items()]
    cohort = {'roles': protocol, 'cohort': rows}
    protocol_path, cohort_path = tmp_path/'protocol.json', tmp_path/'cohort.json'
    follow.write_json_atomic(protocol_path, protocol); follow.write_json_atomic(cohort_path, cohort)
    front_contract = signed(front/'contract.json', {'output_dir': str(cache_path)}, 'contract_sha256')
    identity = signed(job/'contract.json', {'frontend_job': str(front), 'dataset': str(dataset),
        'output_dir': str(train), 'house_protocol': str(protocol_path)}, 'fingerprint')
    checkpoint, report = train/'checkpoint.pt', train/'report.json'
    checkpoint.write_bytes(b'checkpoint'); report.write_text('{}')
    upstream = {'checkpoint': checkpoint, 'report_path': report, 'report': {'selected_step': 1},
                'cache': cache_path, 'diagnostic_contract': {'contract_sha256': 'diag'},
                'frontend_contract': front_contract}
    monkeypatch.setattr(follow, 'wait_completed', lambda *a: (identity, {'status': 'completed'}))
    monkeypatch.setattr(follow, 'diagnostic_inputs', lambda *a: upstream)
    monkeypatch.setattr(follow, 'code_hashes', lambda: {})
    monkeypatch.setattr(aux, 'house_protocol', lambda *a, **k: protocol)
    monkeypatch.setattr(aux, 'validate_cache_house_roles', lambda *a, **k: {})
    floors = [dict(floor_uid='f'+h, observation_order=['a'+h, 'b'+h]) for h in ('h1', 'h2', 'h3')]
    cache = SimpleNamespace(floors=floors, fingerprint='cache_fingerprint', snapshot='cache_snapshot',
                            check_unchanged=lambda: 3)
    monkeypatch.setattr(frozen, 'FrozenOpeningCache', lambda p: cache)
    pair_state = {'training_status': 'trained', 'selected_step': 1,
        'training_contract': {'house_protocol': protocol, 'source_contract': cache.fingerprint,
                              'source_snapshot': cache.snapshot, 'objective': 'known_cells'}}
    monkeypatch.setattr(pipeline, 'FloorModelBundle', lambda *a, **k: SimpleNamespace(pair_state=pair_state))
    monkeypatch.setattr(geometry, 'select_geometry', lambda parts: {'status': 'diagnostic'})
    events, interrupted = [], [False]
    def predict(cache_arg, floor, bundle, output):
        uid = floor['floor_uid']; events.append('predict:'+uid)
        if uid == 'fh2' and not interrupted[0]:
            interrupted[0] = True
            raise KeyboardInterrupt()
        return follow.atomic_artifact(output/'predictions'/(uid+'.json'),
            dict(floor_uid=uid, status='predicted', pair_denominator=1, ground_truth_read=False))
    def hashes(dataset_arg, selected_rows):
        events.append('labels')
        assert (tmp_path/'out/predictions/fh1.json').exists()
        assert (tmp_path/'out/predictions/fh2.json').exists()
        assert [r['role'] for r in selected_rows] == ['fit', 'selection']
        return {}
    def audit(cache_arg, floor, role, house, prediction, dataset_arg):
        events.append('audit:'+floor['floor_uid'])
        return dict(floor_uid=floor['floor_uid'], role=role, status='audited',
                    prediction_fingerprint=prediction['fingerprint'],
                    geometry=dict(oracles=[], inventory=[], rows=[]), selected_pair_outcomes=[])
    monkeypatch.setattr(follow, 'predict_floor', predict)
    monkeypatch.setattr(follow, 'annotation_hashes', hashes)
    monkeypatch.setattr(follow, 'audit_one', audit)
    with pytest.raises(KeyboardInterrupt): follow.run(job, tmp_path/'out')
    assert 'labels' not in events
    report = follow.run(job, tmp_path/'out', resume=True)
    assert report['floor_denominator'] == 2 and report['pair_denominator'] == 2
    assert events.count('predict:fh1') == 1  # first floor reused after interruption
    assert events.index('labels') > events.index('predict:fh2')
    assert follow.read(tmp_path/'out/prediction_barrier.json')['all_predictions_committed_before_labels']
