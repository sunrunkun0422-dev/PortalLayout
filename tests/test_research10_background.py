"""Contract, retry and interruption behaviour for the background runner."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools import research10_background as bg


FAKE_TRAINER = '''\
import argparse, json, pathlib, sys, time
p = argparse.ArgumentParser()
p.add_argument('train'); p.add_argument('--stage'); p.add_argument('--output')
p.add_argument('--dataset'); p.add_argument('--prediction-cache'); p.add_argument('--supervision')
p.add_argument('--mode'); p.add_argument('--seed'); p.add_argument('--device'); p.add_argument('--pair-steps'); p.add_argument('--resume')
p.add_argument('--behavior', default='retry')
a = p.parse_args(); out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
if a.behavior == 'no_checkpoint': sys.exit(9)
if a.behavior == 'nonfinite':
    (out/'last.pt').write_text('checkpoint'); print('nonfinite gradient', flush=True); sys.exit(9)
if a.behavior == 'bad_success': sys.exit(0)
if a.behavior == 'retryforever':
    (out/'last.pt').write_text('checkpoint'); sys.exit(9)
if not a.resume:
    (out/'last.pt').write_text('checkpoint')
    if a.behavior == 'interrupt': time.sleep(60)
    if a.behavior == 'retry': sys.exit(9)
if a.resume and not pathlib.Path(a.resume).is_file(): sys.exit(7)
(out/'checkpoint.pt').write_text('checkpoint')
(out/'training_report.json').write_text(json.dumps({'training_status':'smoke_trained'}))
'''


def spec_for(tmp_path, behavior='retry', retries=2):
    script = tmp_path / 'fake_train.py'
    script.write_text(FAKE_TRAINER)
    dataset, cache, supervision = (tmp_path / 'dataset', tmp_path / 'cache', tmp_path / 'supervision.json')
    dataset.mkdir(); cache.mkdir(); supervision.write_text('{}')
    (dataset / 'data.txt').write_text('dataset-v1'); (cache / 'cache.txt').write_text('cache-v1')
    output, job = tmp_path / 'output', tmp_path / 'job'
    argv = [sys.executable, str(script), 'train', '--stage', 'pairs', '--output', str(output),
            '--dataset', str(dataset), '--prediction-cache', str(cache), '--supervision', str(supervision),
            '--mode', 'smoke', '--seed', '123', '--device', 'cpu', '--pair-steps', '200', '--behavior', behavior]
    spec = {'kind': 'simulation', 'cwd': str(tmp_path), 'argv': argv, 'output_dir': str(output),
            'fingerprint_paths': [str(dataset), str(cache), str(supervision)], 'max_retries': retries}
    spec_file = tmp_path / 'spec.json'; spec_file.write_text(json.dumps(spec))
    return spec, spec_file, job, output


def status(job):
    return json.loads((job / 'status.json').read_text())


def test_failure_retries_from_checkpoint_and_preserves_failure_count(tmp_path):
    _, spec_file, job, output = spec_for(tmp_path)
    bg.prepare(spec_file, job)
    assert bg.run(job) == 0
    result = status(job)
    assert result['status'] == 'completed'
    assert result['attempts'] == 2 and result['failures'] == 1 and result['resumed'] is True
    assert (output / 'training_report.json').is_file()
    assert 'quality gates were not assessed' in result['note']


@pytest.mark.parametrize('behavior,expected', [
    ('no_checkpoint', 'no valid checkpoint'),
    ('nonfinite', 'non-retryable failure: nonfinite'),
    ('bad_success', 'successful exit without required artifacts'),
])
def test_never_treat_failed_or_incomplete_run_as_success(tmp_path, behavior, expected):
    _, spec_file, job, _ = spec_for(tmp_path, behavior)
    bg.prepare(spec_file, job)
    assert bg.run(job) == 2
    assert status(job)['status'] == 'blocked'
    assert expected in status(job)['reason']
    assert status(job)['attempts'] == 1


def test_changed_data_blocks_before_child(tmp_path):
    _, spec_file, job, output = spec_for(tmp_path)
    bg.prepare(spec_file, job)
    (tmp_path / 'dataset' / 'data.txt').write_text('modified')
    assert bg.run(job) == 2
    assert status(job)['status'] == 'blocked'
    assert 'contract source or data changed' in status(job)['reason']
    assert not output.exists()


def test_house_protocol_must_be_fingerprinted(tmp_path):
    spec, _, job, _ = spec_for(tmp_path)
    protocol = tmp_path / 'house_protocol.json'; protocol.write_text('{}')
    spec['argv'] += ['--house-protocol', str(protocol)]
    with pytest.raises(ValueError, match='--house-protocol'):
        bg.build_contract(spec, job)
    spec['fingerprint_paths'].append(str(protocol))
    contract = bg.build_contract(spec, job)
    assert str(protocol.resolve()) in contract['fingerprint_paths']


def test_formal_cannot_use_simulation_or_missing_gate(tmp_path):
    spec, _, job, _ = spec_for(tmp_path)
    spec['purpose'] = 'formal'
    with pytest.raises(ValueError, match='simulation cannot be formal'):
        bg.build_contract(spec, job)


def test_diagnostic_job_rejects_more_than_5000_steps(tmp_path):
    spec, _, job, _ = spec_for(tmp_path)
    spec['argv'][spec['argv'].index('--pair-steps') + 1] = '5001'
    with pytest.raises(ValueError, match='1..5000'):
        bg.build_contract(spec, job)


def test_retries_are_bounded(tmp_path):
    _, spec_file, job, _ = spec_for(tmp_path, 'retryforever', retries=1)
    bg.prepare(spec_file, job)
    assert bg.run(job) == 2
    assert status(job)['status'] == 'blocked'
    assert status(job)['reason'] == 'retry limit exhausted'
    assert status(job)['attempts'] == 2


def test_formal_job_requires_immutable_ready_gate(tmp_path):
    spec, spec_file, job, _ = spec_for(tmp_path)
    spec['kind'], spec['purpose'] = 'floor-train', 'formal'
    spec['argv'][1] = str(bg.ROOT / 'tools' / 'run_floor_layout.py')
    gate = tmp_path / 'gate.json'
    spec['gate_report'] = str(gate)
    gate.write_text(json.dumps({'status': 'blocked', 'rounds': {}}))
    with pytest.raises(ValueError, match='not ready'):
        bg.build_contract(spec, job)
    gate.write_text(json.dumps({'status': 'ready', 'rounds': {'R{:02d}'.format(i): 'passed' for i in range(1, 11)}}))
    spec_file.write_text(json.dumps(spec))
    bg.prepare(spec_file, job)
    gate.write_text('{}')
    assert bg.run(job) == 2
    assert 'contract source or data changed' in status(job)['reason']


def test_signal_interrupt_then_resume_does_not_consume_retry(tmp_path):
    _, spec_file, job, output = spec_for(tmp_path, 'interrupt', retries=0)
    bg.prepare(spec_file, job)
    command = [sys.executable, str(bg.SCRIPT), 'run', '--job-dir', str(job)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        while not (output / 'last.pt').exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(.05)
        assert (output / 'last.pt').exists()
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=10)
        assert proc.returncode == 130, err.decode()
        assert status(job)['status'] == 'interrupted'
        assert status(job)['failures'] == 0
        resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert resumed.returncode == 0, resumed.stderr
        assert status(job)['status'] == 'completed'
        assert status(job)['attempts'] == 2 and status(job)['failures'] == 0
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def test_floor_train_requires_real_entrypoint(tmp_path):
    spec, _, job, _ = spec_for(tmp_path)
    spec['kind'] = 'floor-train'
    with pytest.raises(ValueError, match='entry point'):
        bg.build_contract(spec, job)


FAKE_CACHE = '''\
import argparse, json, pathlib, sys
p = argparse.ArgumentParser()
for name in ('cohort','scope','dataset','raw-root','config','device','batch-size','output','behavior'):
    p.add_argument('--'+name)
p.add_argument('--resume', action='store_true'); a = p.parse_args()
out = pathlib.Path(a.output); counter = out.parent/'count.txt'
attempt = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(attempt+1))
if a.behavior == 'fresh_retry' and attempt == 0: sys.exit(8)
out.mkdir(parents=True, exist_ok=True)
if not a.resume and a.behavior == 'retry':
    (out/'cache_contract.json').write_text('{}'); sys.exit(9)
if a.behavior == 'retry' and not a.resume: sys.exit(6)
if a.resume and not (out/'cache_contract.json').is_file(): sys.exit(7)
(out/'cache_contract.json').write_text('{}')
count = 1 if a.behavior != 'wrong_count' else 2
(out/'run_report.json').write_text(json.dumps({'status':'complete', 'ground_truth_read':False, 'input_errors':[], 'floor_samples':count}))
'''


def cache_spec_for(tmp_path, behavior='retry', scope='development'):
    script = tmp_path / 'fake_cache.py'; script.write_text(FAKE_CACHE)
    dataset, raw = tmp_path / 'dataset', tmp_path / 'raw'
    dataset.mkdir(); raw.mkdir()
    (raw / 'one.jpg').write_text('fit-image'); (raw / 'audit.jpg').write_text('audit-image')
    rows, hashes = [], {}
    for role, image in [('fit', 'one.jpg'), ('audit', 'audit.jpg')]:
        uid = role + '-floor'; rel = role + '/manifest.json'; path = dataset / rel
        path.parent.mkdir(); path.write_text(json.dumps({'floor_uid': uid,
            'observations': [{'rgb': {'root_key': 'zind_raw', 'relative_path': image}}]}))
        rows.append({'role': role, 'floor_uid': uid, 'input_manifest': rel})
        hashes[uid] = hashlib.sha256(path.read_bytes()).hexdigest()
    cohort = tmp_path / 'cohort.json'; cohort.write_text(json.dumps({'schema_version': 'research10.cohort.v1',
        'dataset_root': str(dataset), 'raw_root': str(raw), 'cohort': rows, 'input_manifest_sha256': hashes}))
    model_files = []
    for i in range(3):
        p = tmp_path / ('model' + str(i)); p.write_text('model'); model_files.append(str(p))
    config = tmp_path / 'config.json'; config.write_text(json.dumps(dict(zip(
        ('bi_layout_config', 'bi_layout_checkpoint', 'opening_checkpoint'), model_files))))
    output, job = tmp_path / 'output', tmp_path / 'job'
    argv = [sys.executable, str(script), '--cohort', str(cohort), '--scope', scope,
        '--dataset', str(dataset), '--raw-root', str(raw), '--config', str(config),
        '--device', 'cpu', '--batch-size', '2', '--output', str(output), '--behavior', behavior]
    spec = {'kind': 'simulation-cache', 'cwd': str(tmp_path), 'argv': argv,
        'output_dir': str(output), 'max_retries': 2}
    spec_file = tmp_path / 'spec.json'; spec_file.write_text(json.dumps(spec))
    return spec, spec_file, job, output


def test_frontend_cache_retry_is_flag_only_and_scope_binds_inputs(tmp_path):
    _, spec_file, job, output = cache_spec_for(tmp_path)
    contract = bg.prepare(spec_file, job)
    assert contract['cache_scope'] == 'development' and contract['cache_floor_count'] == 1
    paths = contract['fingerprint_paths']
    assert str((tmp_path/'dataset'/'audit'/'manifest.json').resolve()) in paths
    assert str((tmp_path/'raw'/'one.jpg').resolve()) in paths
    assert str((tmp_path/'raw'/'audit.jpg').resolve()) not in paths
    assert bg.run(job) == 0
    assert status(job)['status'] == 'completed' and status(job)['resumed']
    assert (output/'run_report.json').is_file()


def test_frontend_retry_without_any_output_starts_fresh(tmp_path):
    _, spec_file, job, _ = cache_spec_for(tmp_path, 'fresh_retry')
    bg.prepare(spec_file, job)
    assert bg.run(job) == 0
    assert status(job)['attempts'] == 2 and not status(job)['resumed']


def test_frontend_wrong_floor_count_is_blocked(tmp_path):
    _, spec_file, job, _ = cache_spec_for(tmp_path, 'wrong_count')
    bg.prepare(spec_file, job)
    assert bg.run(job) == 2
    assert status(job)['status'] == 'blocked'
    assert 'floor count' in status(job)['reason']


def test_frontend_manifest_or_image_changes_block_before_execution(tmp_path):
    _, spec_file, job, _ = cache_spec_for(tmp_path)
    bg.prepare(spec_file, job)
    (tmp_path/'raw'/'one.jpg').write_text('different')
    assert bg.run(job) == 2
    assert 'contract source or data changed' in status(job)['reason']


def test_frontend_cache_rejects_changed_manifest_hash(tmp_path):
    spec, _, job, _ = cache_spec_for(tmp_path)
    (tmp_path/'dataset'/'audit'/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        bg.build_contract(spec, job)
