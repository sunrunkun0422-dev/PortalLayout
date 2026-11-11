import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools import research10_pipeline as pipe


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(tmp_path):
    root = tmp_path/'source'; dataset = root/'dataset'; dataset.mkdir(parents=True)
    roles = {role: [str(i)] for i, role in enumerate(('fit', 'selection', 'calibration', 'audit'), 1)}
    rows = [{'role': k, 'house_id': v[0], 'floor_uid': 'floor-'+k} for k, v in roles.items()]
    cohort = root/'cohort.json'
    write(cohort, {'schema_version':'research10.cohort.v1', 'dataset_root':str(dataset), 'roles':roles, 'cohort':rows})
    protocol = root/'house_protocol.json'
    value = dict(schema_version='research10.house-protocol.v1', seed=123,
                 source_sha256=pipe.bg.file_hash(cohort), **roles)
    value['fingerprint'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write(protocol, value)
    write(dataset/'splits'/'official_house_split.json', {'train':['1','2','3','4'], 'val':[], 'test':[]})
    front = root/'frontend_job'; front.mkdir()
    frontcontract = {'schema':pipe.bg.SCHEMA, 'kind':'frontend-cache', 'cache_scope':'development',
        'cwd':str(tmp_path), 'argv':['python','--cohort',str(cohort),'--dataset',str(dataset)],
        'output_dir':str(root/'cache'), 'fingerprint_paths':{}}
    frontcontract['contract_sha256'] = hashlib.sha256(pipe.bg.canonical(frontcontract)).hexdigest()
    write(front/'contract.json', frontcontract)
    write(front/'status.json', {'status':'completed','contract_sha256':frontcontract['contract_sha256']})
    write(root/'cache'/'run_report.json', {'status':'complete','scope':'development',
        'ground_truth_read':False,'input_errors':[], 'floor_samples':3,
        'floors':[{'floor_uid':r['floor_uid']} for r in rows if r['role']!='audit']})
    output, job = tmp_path/'result', tmp_path/'pipeline'
    spec = {'frontend_job':str(front),'dataset':str(dataset),'house_protocol':str(protocol),
            'output_dir':str(output),'python':sys.executable}
    specfile = tmp_path/'spec.json'; write(specfile, spec)
    return specfile, job, output, front, roles


def test_frontend_exact_development_membership(tmp_path):
    specfile, job, _, front, _ = fixture(tmp_path)
    identity = pipe.prepare(specfile, job)
    assert pipe._front(identity)['cache_scope'] == 'development'
    report = pipe.bg.load_json(Path(pipe.bg.load_json(front/'contract.json')['output_dir'])/'run_report.json')
    report['floors'][0]['floor_uid'] = 'floor-audit'
    write(Path(pipe.bg.load_json(front/'contract.json')['output_dir'])/'run_report.json', report)
    with pytest.raises(ValueError, match='exactly cover'):
        pipe._front(identity)


def test_overlap_roles_rejected(tmp_path):
    specfile, job, _, _, _ = fixture(tmp_path)
    spec = pipe.bg.load_json(specfile)
    protocol = Path(spec['house_protocol']); raw = pipe.bg.load_json(protocol)
    raw['audit'] = raw['fit']; write(protocol, raw)
    with pytest.raises(ValueError, match='roles differ'):
        pipe.prepare(specfile, job)


def test_supervision_rejects_audit_and_missing_identity(tmp_path):
    specfile, job, _, _, roles = fixture(tmp_path)
    identity = pipe.prepare(specfile, job)
    protocol = pipe.bg.load_json(identity['house_protocol'])
    weak = tmp_path/'weak.json'
    def save(row):
        d = {'schema_version':'opening-pair-supervision.v1','usage':'training_calibration',
             'house_protocol':protocol, 'pairs':[row]}
        d['fingerprint'] = hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
        write(weak,d)
    save({'floor_uid':'floor-audit','house_id':roles['audit'][0]})
    with pytest.raises(ValueError, match='audit, heldout'): pipe._weak(identity, weak)
    save({'floor_uid':'missing','house_id':None})
    with pytest.raises(ValueError, match='audit, heldout'): pipe._weak(identity, weak)
    save({'floor_uid':'floor-fit','house_id':roles['fit'][0]})
    assert len(pipe._weak(identity, weak)['pairs']) == 1


def test_smoke_and_diagnostic_specs_are_bounded_and_use_same_protocol(tmp_path):
    specfile, job, _, _, _ = fixture(tmp_path)
    identity = pipe.prepare(specfile, job)
    for diagnostic, steps, mode in ((False,'200','smoke'),(True,'5000','full')):
        spec = pipe._training_spec(identity, tmp_path/'cache', tmp_path/'weak', diagnostic)
        assert spec['purpose']=='diagnostic'
        assert pipe.bg._option(spec['argv'], '--pair-steps') == steps
        assert pipe.bg._option(spec['argv'], '--mode') == mode
        assert pipe.bg._option(spec['argv'], '--house-protocol') == identity['house_protocol']
        assert pipe.bg._option(spec['argv'], '--pair-objective') == 'known_cells'
        assert pipe.bg._option(spec['argv'], '--selection-metric') == 'cross_room_ap'


def fake_report(steps, skipped=0):
    return {'training_status':'smoke_trained' if steps==200 else 'trained','steps':steps,
        'parameters_changed':True,'skipped_batches':skipped,
        'gradient_norms_first_step':{'encoder':1.,'matcher':2.}, 'status':'diagnostic_loss_not_decreased'}


@pytest.mark.parametrize('skipped,valid', [(0,True),(1,True),(199,True),(200,False)])
def test_smoke_skips_and_raw_loss_preserved(tmp_path, skipped, valid):
    reportpath = tmp_path/'smoke_pairs'/'training_report.json'
    write(reportpath, fake_report(200,skipped))
    if valid: assert pipe._check_smoke({'output_dir':str(tmp_path)})['status'] == 'diagnostic_loss_not_decreased'
    else:
        with pytest.raises(ValueError): pipe._check_smoke({'output_dir':str(tmp_path)})


def test_invalid_gradient_rejected(tmp_path):
    report = fake_report(200); report['gradient_norms_first_step']['matcher'] = float('nan')
    write(tmp_path/'smoke_pairs'/'training_report.json', report)
    with pytest.raises(ValueError): pipe._check_smoke({'output_dir':str(tmp_path)})


def test_failure_does_not_launch_downstream(tmp_path, monkeypatch):
    specfile, job, output, front, _ = fixture(tmp_path)
    identity = pipe.prepare(specfile, job)
    smokejob = job/'smoke_job'; smokejob.mkdir()
    write(smokejob/'status.json', {'status':'blocked','reason':'synthetic failure'})
    state = pipe.bg.load_json(job/'status.json'); state.update(stage='waiting_smoke', status='running')
    write(job/'status.json',state)
    monkeypatch.setattr(pipe, '_check', lambda *_: None)
    assert pipe.run(job, poll_seconds=0) == 2
    assert pipe.bg.load_json(job/'status.json')['status'] == 'blocked'
    assert not (job/'diagnostic_job').exists()


def test_sequential_workflow_runs_only_bounded_diagnostic(tmp_path, monkeypatch):
    specfile, job, output, front, _ = fixture(tmp_path)
    pipe.prepare(specfile, job)
    monkeypatch.setattr(pipe, '_check', lambda *_: None)
    monkeypatch.setattr(pipe, '_front', lambda *_: {})
    monkeypatch.setattr(pipe, '_weak', lambda *_: {})
    monkeypatch.setattr(pipe.bg, 'check_contract', lambda *_: None)
    monkeypatch.setattr(pipe, '_active', lambda *_: False)
    calls = []
    def prepare_process(cmd, **kwargs):
        directory = Path(cmd[cmd.index('--output')+1]); directory.mkdir(parents=True)
        write(directory/'opening_weak.json', {'synthetic':True})
        return subprocess.CompletedProcess(cmd, 0)
    def create_job(specfile, subjob):
        subjob.mkdir(parents=True); spec = pipe.bg.load_json(specfile)
        calls.append(('prepare', subjob.name, spec['purpose'], pipe.bg._option(spec['argv'],'--pair-steps')))
        write(subjob/'contract.json', {'contract_sha256':'test'})
        write(subjob/'status.json', {'status':'prepared','contract_sha256':'test'})
    def launch(subjob):
        calls.append(('launch', subjob.name))
        steps = 200 if subjob.name == 'smoke_job' else 5000
        name = 'smoke_pairs' if steps == 200 else 'diagnostic_pairs'
        write(output/name/'training_report.json', fake_report(steps, 1))
        write(subjob/'status.json', {'status':'completed','contract_sha256':'test'})
    monkeypatch.setattr(pipe.subprocess, 'run', prepare_process)
    monkeypatch.setattr(pipe.bg, 'prepare', create_job)
    monkeypatch.setattr(pipe.bg, 'launch', launch)
    assert pipe.run(job,poll_seconds=0) == 0
    assert calls == [('prepare','smoke_job','diagnostic','200'),('launch','smoke_job'),
                     ('prepare','diagnostic_job','diagnostic','5000'),('launch','diagnostic_job')]
    state = pipe.bg.load_json(job/'status.json')
    assert state['status']=='completed' and state['diagnostic_steps']==5000
    assert 'formal-training gates not established' in state['note']
    assert (job/'weak_prepare_completion.json').is_file()
    assert not (output/'weak_supervision'/'run_report.json').exists()


def test_interrupted_weak_prepare_without_marker_blocks(tmp_path, monkeypatch):
    specfile, job, output, _, _ = fixture(tmp_path)
    pipe.prepare(specfile, job)
    write(output/'weak_supervision'/'opening_weak.json', {'partial':True})
    state = pipe.bg.load_json(job/'status.json'); state.update(stage='preparing', status='running')
    write(job/'status.json', state)
    monkeypatch.setattr(pipe, '_check', lambda *_: None)
    assert pipe.run(job, poll_seconds=0) == 2
    state = pipe.bg.load_json(job/'status.json')
    assert state['status'] == 'blocked' and 'completion marker missing' in state['reason']
    assert not (job/'smoke_job').exists()


def test_real_prepare_cli_contract_writes_only_weak_output(tmp_path, monkeypatch):
    """Exercise the actual prepare parser/branch with mocked cache and label builder."""
    from tools import run_floor_layout
    import portallayout.utils.frozen_opening_cache as frozen
    import portallayout.datasets.floor_auxiliary_supervision as supervision
    import portallayout.datasets.opening_pair_supervision as opening
    cache = object()
    monkeypatch.setattr(frozen, 'FrozenOpeningCache', lambda _: cache)
    monkeypatch.setattr(supervision, 'house_protocol', lambda *_args, **_kwargs: {'fit':['1']})
    monkeypatch.setattr(supervision, 'validate_cache_house_roles', lambda *_args, **_kwargs: None)
    def build(_cache, _dataset, path, **kwargs):
        write(Path(path), {'real_cli_branch':True})
        return {'statistics':{'pairs':1}}
    monkeypatch.setattr(opening, 'build_weak_supervision', build)
    out = tmp_path/'prepare_result'
    assert run_floor_layout.main(['prepare','--stage','opening','--dataset',str(tmp_path/'dataset'),
        '--prediction-cache',str(tmp_path/'cache'),'--house-protocol',str(tmp_path/'protocol.json'),
        '--output',str(out),'--device','cpu']) == 0
    assert pipe.bg.load_json(out/'opening_weak.json') == {'real_cli_branch':True}
    assert not (out/'run_report.json').exists()
