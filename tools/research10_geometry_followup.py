#!/usr/bin/env python3
"""Resume-safe R02/R03 geometry follow-up for the completed diagnostic job.

Waits for the upstream pipeline. Predictions for ALL fixed fit/selection floors
are committed under inference_guard before any annotation is opened. Offline
audit then reads only those floors' labels. No service is started by this tool.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic

SCHEMA = 'research10.geometry-followup.v1'


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def checked_contract(path, key):
    data = read(path); claimed = data.get(key)
    actual = hashlib.sha256(canonical({k: v for k, v in data.items() if k != key})).hexdigest()
    if not claimed or actual != claimed: raise ValueError('invalid upstream contract fingerprint: ' + str(path))
    return data


def wait_completed(job, poll_seconds=10):
    if poll_seconds <= 0: raise ValueError('poll interval must be positive')
    job = Path(job)
    contract = checked_contract(job/'contract.json', 'fingerprint')
    while True:
        state = read(job/'status.json')
        if state.get('contract_sha256') != contract['fingerprint']:
            raise ValueError('pipeline status/contract mismatch')
        if state.get('status') == 'completed':
            if state.get('stage') != 'done': raise ValueError('completed pipeline has invalid stage')
            return contract, state
        if state.get('status') in ('blocked', 'failed', 'interrupted'):
            raise ValueError('upstream pipeline did not complete: ' + str(state.get('status')))
        if state.get('status') not in ('prepared', 'running', 'retrying'):
            raise ValueError('unknown upstream status: ' + str(state.get('status')))
        print('waiting for diagnostic pipeline: {} / {}'.format(state.get('stage'), state.get('status')), flush=True)
        time.sleep(poll_seconds)


def diagnostic_inputs(job, pipeline, pipeline_state):
    """Validate immutable job metadata without reading any annotation tree."""
    job = Path(job)
    diagnostic = checked_contract(job/'diagnostic_job/contract.json', 'contract_sha256')
    state = read(job/'diagnostic_job/status.json')
    target = Path(pipeline['output_dir'])/'diagnostic_pairs'
    if (state.get('status') != 'completed' or state.get('exit_code') != 0 or
            state.get('contract_sha256') != diagnostic['contract_sha256'] or
            diagnostic.get('kind') != 'floor-train' or diagnostic.get('purpose') != 'diagnostic' or
            diagnostic.get('stage') != 'pairs' or
            Path(diagnostic['output_dir']).resolve() != target.resolve()):
        raise ValueError('diagnostic job status or output contract invalid')
    report_path, checkpoint = target/'training_report.json', target/'checkpoint.pt'
    if content_hash(report_path) != pipeline_state.get('diagnostic_report_sha256'):
        raise ValueError('diagnostic report differs from completed pipeline')
    report = read(report_path); gradients = report.get('gradient_norms_first_step', {})
    if (report.get('training_status') != 'trained' or report.get('parameters_changed') is not True or
            not isinstance(report.get('steps'), int) or not 1 <= report['steps'] <= 5000 or
            not isinstance(report.get('selected_step'), int) or not 1 <= report['selected_step'] <= report['steps'] or
            not isinstance(report.get('skipped_batches'), int) or not 0 <= report['skipped_batches'] < report['steps'] or
            any(not isinstance(gradients.get(k), (int, float)) or not math.isfinite(gradients[k]) or
                gradients[k] <= 0 for k in ('encoder', 'matcher'))):
        raise ValueError('diagnostic checkpoint is not supported by a valid training report')
    if not checkpoint.is_file(): raise ValueError('diagnostic checkpoint missing')
    frontend_job = Path(pipeline['frontend_job'])
    frontend = checked_contract(frontend_job/'contract.json', 'contract_sha256')
    front_state = read(frontend_job/'status.json')
    if (frontend.get('contract_sha256') != pipeline.get('frontend_contract_sha256') or
            front_state.get('contract_sha256') != frontend.get('contract_sha256') or
            front_state.get('status') != 'completed' or frontend.get('cache_scope') != 'development'):
        raise ValueError('development frontend job is not complete under the pipeline contract')
    return {'checkpoint': checkpoint.resolve(), 'report': report, 'report_path': report_path.resolve(),
            'cache': Path(frontend['output_dir']).resolve(), 'diagnostic_contract': diagnostic,
            'frontend_contract': frontend}


def cohort_scope(cohort, protocol):
    roles = ('fit', 'selection', 'calibration', 'audit')
    if any(cohort.get('roles', {}).get(k) != protocol.get(k) for k in roles):
        raise ValueError('cohort and protocol differ')
    rows = cohort.get('cohort', [])
    if any(not isinstance(r.get('floor_uid'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', r['floor_uid']) for r in rows):
        raise ValueError('invalid cohort floor identity')
    if len({r['floor_uid'] for r in rows}) != len(rows): raise ValueError('duplicate cohort floor')
    if len({r['house_id'] for r in rows}) != len(rows): raise ValueError('duplicate cohort house')
    selected = [r for r in rows if r['role'] in ('fit', 'selection')]
    if {r['house_id'] for r in selected} != set(protocol['fit']) | set(protocol['selection']):
        raise ValueError('incomplete fit/selection floor mapping')
    if any(r['house_id'] not in protocol[r['role']] for r in rows):
        raise ValueError('cohort row assigned to wrong role')
    return sorted(selected, key=lambda r: (r['role'], r['floor_uid']))


def checked_artifact(path):
    data = read(path); expected = data.pop('fingerprint', None)
    if expected is None or fingerprint(data) != expected: raise ValueError('artifact fingerprint mismatch: '+str(path))
    data['fingerprint'] = expected
    return data


def atomic_artifact(path, data):
    payload = dict(data); payload['fingerprint'] = fingerprint(payload)
    write_json_atomic(path, payload)
    return payload


def code_hashes():
    files = sorted((ROOT/'portallayout').rglob('*.py'))
    files += [Path(__file__).resolve(), ROOT/'tools/validate_portal_geometry_hypothesis.py']
    return {str(p.resolve()): content_hash(p) for p in files}


def verify_file_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name)
        actual = content_hash(path) if path.is_file() else None
        if actual != expected: raise ValueError('frozen source changed: ' + name)


def annotation_hashes(dataset, rows):
    """Only these three files in each fixed fit/selection floor may be read."""
    result = {}
    for row in rows:
        if row['role'] not in ('fit', 'selection'): raise ValueError('forbidden annotation role')
        for name in ('floor_gt.json', 'observations_gt.jsonl', 'portal_links.jsonl'):
            path = Path(dataset)/'labels'/row['floor_uid']/name
            if path.resolve() != path.absolute(): raise ValueError('annotation path traverses a symlink')
            result[str(path)] = content_hash(path) if path.is_file() else None
    return result


def create_contract(job, identity, upstream, protocol_path, cohort_path, cache, rows, device):
    return {'schema_version': SCHEMA, 'pipeline_contract_sha256': identity['fingerprint'],
        'diagnostic_job_contract_sha256': upstream['diagnostic_contract']['contract_sha256'],
        'frontend_job_contract_sha256': upstream['frontend_contract']['contract_sha256'],
        'pipeline_job': str(Path(job).resolve()), 'dataset': str(Path(identity['dataset']).resolve()),
        'protocol_path': str(Path(protocol_path).resolve()), 'protocol_sha256': content_hash(protocol_path),
        'cohort_path': str(Path(cohort_path).resolve()), 'cohort_sha256': content_hash(cohort_path),
        'checkpoint': str(upstream['checkpoint']), 'checkpoint_sha256': content_hash(upstream['checkpoint']),
        'training_report': str(upstream['report_path']), 'training_report_sha256': content_hash(upstream['report_path']),
        'cache': str(upstream['cache']), 'cache_fingerprint': cache.fingerprint, 'cache_snapshot': cache.snapshot,
        'code_sha256': code_hashes(), 'device': device, 'floors': rows, 'opening_threshold': .7,
        'annotation_scope': ['fit', 'selection'], 'source_tree_rehashed': False,
        'source_tree_note': 'Full upstream dataset tree is not rehashed because that would read protected labels.'}


def compare_contract(path, contract):
    if path.exists():
        saved = checked_artifact(path)
        if {k: v for k, v in saved.items() if k != 'fingerprint'} != contract:
            raise ValueError('resume contract differs; use a new output directory')
    else: atomic_artifact(path, contract)


def predict_floor(cache, floor, bundle, output):
    from portallayout.engine.floor_layout_pipeline import exploratory_pair_predictions
    from portallayout.geometry.exploratory_floor_pose import select_exploratory_poses
    from portallayout.geometry.floor_pose_graph import PoseGraphConfig
    from portallayout.geometry.pose_geometry import METHODS, opening_overlay
    from portallayout.utils.pose_inference_guard import inference_guard
    uid = floor['floor_uid']; pair_dir = output/'pair_predictions'
    with inference_guard():
        observations, raw, timing = exploratory_pair_predictions(floor, cache, bundle,
                                                                 {'pair_prediction_cache': str(pair_dir)})
        expected = list(itertools.combinations(sorted(floor['observation_order']), 2))
        if [(x['observation_a'], x['observation_b']) for x in raw] != expected:
            raise ValueError('prediction does not cover all unordered pairs')
        selected = select_exploratory_poses(raw, observations, .7, PoseGraphConfig())
        overlays = {m: {} for m in METHODS}
        for obs in observations:
            oid = obs['observation_id']
            if oid not in cache.records:
                for method in METHODS: overlays[method][oid] = obs
                continue
            record, arrays = cache.load(oid)
            for method in METHODS: overlays[method][oid] = opening_overlay(record, arrays, method)
        data = {'schema_version': 'research10.geometry-prediction.v1', 'floor_uid': uid,
                'status': 'predicted', 'ground_truth_read': False, 'pair_denominator': len(expected),
                'observations': observations, 'raw_pairs': raw, 'selected_pairs': selected,
                'overlays': overlays, 'timing': timing, 'selection': 'exploratory_uncalibrated_threshold_0.7'}
        return atomic_artifact(output/'predictions'/(uid+'.json'), data)


def predict_marker(cache, floor, bundle, output):
    uid = floor['floor_uid']; path = output/'predictions'/(uid+'.json')
    if path.exists():
        data = checked_artifact(path)
        if data.get('floor_uid') != uid: raise ValueError('prediction floor identity mismatch')
        return data
    try: return predict_floor(cache, floor, bundle, output)
    except Exception as error:
        return atomic_artifact(path, {'schema_version': 'research10.geometry-prediction.v1',
            'floor_uid': uid, 'status': 'failed', 'ground_truth_read': False,
            'pair_denominator': len(floor['observation_order'])*(len(floor['observation_order'])-1)//2,
            'error_type': type(error).__name__, 'reason': str(error)})


def audit_one(cache, floor, role, expected_house, prediction, dataset):
    from portallayout.datasets.zind_floor_dataset import read_jsonl
    from portallayout.evaluation.floor_metrics import load_floor_ground_truth
    from portallayout.evaluation.pose_geometry import audit_floor, truth_for, CONFIGS
    from portallayout.evaluation.pose_candidate_audit import audit_floor as selected_audit
    from portallayout.datasets.floor_auxiliary_supervision import pose_target
    from portallayout.geometry.pose_geometry import pose_hypotheses, variant_id
    base = Path(dataset)/'labels'/floor['floor_uid']
    if read(base/'floor_gt.json').get('house_id') != expected_house:
        raise ValueError('annotation house differs from fixed cohort')
    gt = load_floor_ground_truth(dataset, floor['floor_uid'])
    annotations = {r['observation_id']: r for r in read_jsonl(base/'observations_gt.jsonl')}
    links = list(read_jsonl(base/'portal_links.jsonl'))
    geometry = audit_floor(cache, floor, prediction['raw_pairs'], dataset, role, prediction['overlays'])
    # audit_floor's weak groups store only a success boolean. Reconstruct the
    # original per-hypothesis labels so grey-zone (-1) is never called wrong.
    raw_by_id = {r['pair_id']: r for r in prediction['raw_pairs']}
    for row in geometry['rows']:
        if row['collection'] != 'weak': continue
        pair = raw_by_id[row['pair_id']]
        a, b = pair['observation_a'], pair['observation_b']
        truth = truth_for(gt, a, b)
        for config in CONFIGS:
            overlay = prediction['overlays'][config['pose_geometry']]
            openings_a = {o['candidate_id']: o for o in overlay[a]['prediction']['openings']}
            openings_b = {o['candidate_id']: o for o in overlay[b]['prediction']['openings']}
            labels = []
            for ca, cb in row['seed_candidate_ids']:
                try:
                    hypotheses = pose_hypotheses(openings_a[ca]['endpoints_local'], openings_b[cb]['endpoints_local'],
                                                 config['scale_shrinkage'])
                except (ValueError, TypeError, KeyError): hypotheses = []
                labels.extend(pose_target(h['T_B_to_A'], truth)[0] for h in hypotheses)
            row['variants'][variant_id(config)]['labels'] = labels
    for oracle in geometry['oracles']:
        oracle['association_or_annotation_status'] = classify_labels(
            oracle['conditions']['O_full']['labels'], oracle['gt_valid'])
        oracle['legacy_inconsistent_flag_scope'] = 'legacy flag also includes grey-zone and unavailable cases'
    selected = selected_audit(gt, annotations, links, cache.records, prediction['observations'],
                              prediction['raw_pairs'], prediction['selected_pairs'])
    selected_rows = []
    for pair in prediction['selected_pairs']:
        a, b = pair['observation_a'], pair['observation_b']
        truth = truth_for(gt, a, b)
        transform = pair.get('T_B_to_A')
        if transform is None: outcome = 'no_pose'
        elif truth is None: outcome = 'gt_unavailable'
        else: outcome = {1: 'correct', 0: 'incorrect', -1: 'unknown'}[pose_target(transform, truth)[0]]
        selected_rows.append({'pair_id': pair['pair_id'], 'outcome': outcome,
                              'pose_status': pair.get('pose_status'), 'calibrated': pair.get('calibrated')})
    return {'schema_version': 'research10.geometry-floor-audit.v1', 'floor_uid': floor['floor_uid'],
            'role': role, 'status': 'audited', 'prediction_fingerprint': prediction['fingerprint'],
            'geometry': geometry, 'selected_pose_audit': selected, 'selected_pair_outcomes': selected_rows,
            'note': 'candidate upper bounds and GT/weak-label oracles are offline diagnostics; '
                    'selected_geometry is uncalibrated exploratory selection, not verified edges'}


def classify_labels(labels, gt_valid=True):
    if not gt_valid: return 'gt_unavailable'
    if not labels: return 'no_hypotheses'
    if any(label not in (-1, 0, 1) for label in labels): raise ValueError('invalid pose_target label')
    if 1 in labels: return 'solvable'
    if -1 in labels: return 'unknown'
    return 'all_incorrect'


def run(job, output, device='cpu', poll_seconds=10, resume=False):
    from portallayout.datasets.floor_auxiliary_supervision import house_protocol, validate_cache_house_roles
    from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
    from portallayout.engine.floor_layout_pipeline import FloorModelBundle
    import torch
    job, output = Path(job).resolve(), Path(output).resolve()
    preliminary = checked_contract(job/'contract.json', 'fingerprint')
    front = checked_contract(Path(preliminary['frontend_job'])/'contract.json', 'contract_sha256')
    protected = [job, Path(preliminary['dataset']).resolve(), Path(preliminary['output_dir']).resolve(),
                 Path(front['output_dir']).resolve()]
    if any(output == path or path in output.parents for path in protected):
        raise ValueError('output overlaps upstream job, dataset, training or prediction cache')
    if output.exists() and any(output.iterdir()) and not resume: raise ValueError('use new output or --resume')
    if not output.exists() and resume: raise ValueError('cannot resume missing output')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'followup.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def state(**value): write_json_atomic(output/'status.json', dict(schema_version=SCHEMA, **value))
        try:
            compare_contract(output/'request.json', {'schema_version': SCHEMA, 'pipeline_job': str(job),
                                                       'device': device, 'script_sha256': content_hash(__file__)})
            state(stage='waiting_pipeline', status='waiting')
            identity, upstream_state = wait_completed(job, poll_seconds)
            upstream = diagnostic_inputs(job, identity, upstream_state)
            dataset = Path(identity['dataset']).resolve()
            protocol_path = Path(identity['house_protocol']).resolve(); cohort_path = protocol_path.parent/'cohort.json'
            protocol = house_protocol(dataset, protocol_path=protocol_path); cohort = read(cohort_path)
            rows = cohort_scope(cohort, protocol)
            torch.set_num_threads(2)
            cache = FrozenOpeningCache(upstream['cache'])
            validate_cache_house_roles(cache, dataset, protocol, {'fit', 'selection', 'calibration'}, cohort_path)
            floor_map = {r['floor_uid']: r for r in cache.floors}
            expected_dev = {r['floor_uid'] for r in cohort['cohort'] if r['role'] in ('fit', 'selection', 'calibration')}
            if set(floor_map) != expected_dev: raise ValueError('cache does not exactly cover development cohort')
            bundle = FloorModelBundle({'pair_checkpoint': str(upstream['checkpoint'])}, device)
            checkpoint = bundle.pair_state
            train = checkpoint.get('training_contract', {})
            if (checkpoint.get('training_status') != 'trained' or checkpoint.get('selected_step') != upstream['report']['selected_step'] or
                    train.get('house_protocol') != protocol or train.get('source_contract') != cache.fingerprint or
                    train.get('source_snapshot') != cache.snapshot or train.get('objective') != 'known_cells'):
                raise ValueError('diagnostic checkpoint does not match report, cache, or protocol')
            contract = create_contract(job, identity, upstream, protocol_path, cohort_path, cache, rows, device)
            compare_contract(output/'run_contract.json', contract)
            verify_file_hashes(contract['code_sha256'])
            predictions = []
            for index, row in enumerate(rows):
                state(stage='prediction', status='running', completed_floors=index, total_floors=len(rows), floor_uid=row['floor_uid'])
                predictions.append(predict_marker(cache, floor_map[row['floor_uid']], bundle, output))
            # This barrier is explicit: no annotation hashing or reads above it.
            barrier = {'schema_version': SCHEMA, 'prediction_count': len(predictions),
                       'floor_uids': [p['floor_uid'] for p in predictions],
                       'prediction_hashes': {p['floor_uid']: content_hash(output/'predictions'/(p['floor_uid']+'.json')) for p in predictions},
                       'all_predictions_committed_before_labels': True}
            compare_contract(output/'prediction_barrier.json', barrier)
            annotation = {'schema_version': SCHEMA, 'annotation_scope': ['fit', 'selection'],
                          'files': annotation_hashes(dataset, rows)}
            compare_contract(output/'annotation_contract.json', annotation)
            audits = []
            for index, (row, prediction) in enumerate(zip(rows, predictions)):
                uid = row['floor_uid']; path = output/'audits'/(uid+'.json')
                state(stage='offline_audit', status='running', completed_floors=index, total_floors=len(rows), floor_uid=uid)
                if path.exists():
                    audit = checked_artifact(path)
                    if audit.get('prediction_fingerprint') != prediction['fingerprint']:
                        raise ValueError('audit is bound to a different floor prediction')
                else:
                    try:
                        if prediction['status'] != 'predicted': raise ValueError('prediction_failed')
                        audit = audit_one(cache, floor_map[uid], row['role'], row['house_id'], prediction, dataset)
                    except Exception as error:
                        audit = {'schema_version': 'research10.geometry-floor-audit.v1', 'floor_uid': uid,
                                 'role': row['role'], 'status': 'failed', 'prediction_fingerprint': prediction['fingerprint'],
                                 'error_type': type(error).__name__, 'reason': str(error)}
                    audit = atomic_artifact(path, audit)
                audits.append(audit)
            verify_file_hashes(annotation['files']); verify_file_hashes(contract['code_sha256'])
            for name in ('protocol_path', 'cohort_path', 'checkpoint', 'training_report'):
                if content_hash(contract[name]) != contract[name.replace('_path', '')+'_sha256' if name.endswith('_path') else name+'_sha256']:
                    raise ValueError('contract source changed: ' + name)
            cache.check_unchanged()
            summary = summarize_followup(rows, predictions, audits)
            summary['run_contract_fingerprint'] = checked_artifact(output/'run_contract.json')['fingerprint']
            summary['annotation_contract_fingerprint'] = checked_artifact(output/'annotation_contract.json')['fingerprint']
            atomic_artifact(output/'summary.json', summary)
            state(stage='finished', status='completed_diagnostic_only',
                  prediction_failures=summary['prediction_failures'], audit_failures=summary['audit_failures'])
            return summary
        except BaseException as error:
            state(stage='stopped', status='interrupted' if isinstance(error, KeyboardInterrupt) else 'blocked',
                  error_type=type(error).__name__, reason=str(error))
            raise


def summarize_followup(rows, predictions, audits):
    if len(rows) != len(predictions) or len(rows) != len(audits): raise ValueError('floor accounting mismatch')
    if any(r['floor_uid'] != p['floor_uid'] or r['floor_uid'] != a['floor_uid']
           for r, p, a in zip(rows, predictions, audits)): raise ValueError('floor identity mismatch')
    parts = [a['geometry'] for a in audits if a['status'] == 'audited']
    from portallayout.evaluation.pose_geometry import select_geometry
    selection = select_geometry(parts) if parts else None
    categories, variant_outcomes, selected_outcomes = Counter(), {}, Counter()
    for part in parts:
        for oracle in part['oracles']:
            status = oracle['association_or_annotation_status']
            categories['oracle_full_'+status] += 1
        for row in part['inventory']:
            categories['pair_inventory'] += 1
            categories['unknown_room_pairs'] += row['relationship'] == 'unknown_room'
            categories['invalid_gt_pairs'] += not row['gt_valid']
        categories['retrieved_pair_rows'] += sum(row['collection'] == 'r3' for row in part['rows'])
        categories['weak_correspondence_pair_rows'] += sum(row['collection'] == 'weak' for row in part['rows'])
        for row in part['rows']:
            for variant, info in row['variants'].items():
                key = '/'.join((row['role'], row['collection'], row['relationship'], variant))
                counts = variant_outcomes.setdefault(key, Counter())
                counts[classify_labels(info['labels'], row['gt_valid'])] += 1
    for audit in audits:
        if audit['status'] == 'audited':
            selected_outcomes.update(row['outcome'] for row in audit['selected_pair_outcomes'])
    return {'schema_version': SCHEMA, 'status': 'diagnostic_only', 'floor_denominator': len(rows),
            'pair_denominator': sum(p['pair_denominator'] for p in predictions),
            'prediction_failures': sum(p['status'] != 'predicted' for p in predictions),
            'prediction_failed_pair_denominator': sum(p['pair_denominator'] for p in predictions if p['status'] != 'predicted'),
            'audit_failures': sum(a['status'] != 'audited' for a in audits),
            'audit_failed_pair_denominator': sum(p['pair_denominator'] for p, a in zip(predictions, audits)
                                                 if a['status'] != 'audited'),
            'audit_failed_floor_uids': [a['floor_uid'] for a in audits if a['status'] != 'audited'],
            'counts_on_successfully_audited_floors': dict(categories),
            'variant_outcomes': {key: dict(value) for key, value in sorted(variant_outcomes.items())},
            'selected_pair_outcomes': dict(selected_outcomes), 'geometry_selection_diagnostic': selection,
            'limitations': ['No audit/calibration/test annotations read by this follow-up.',
                '3x3 candidate success is an offline pose-oracle upper bound, not selected edge correctness.',
                'Weak correspondences are geometric weak labels, not human-verified.',
                'Candidate-success denominators include unknowns; unknowns are separately counted, not wrong.',
                'Legacy oracle inconsistency boolean can include grey-zone; use oracle_full_* statuses.',
                'The inherited r3 collection key denotes the NEW diagnostic matcher here, not historic R3 weights.',
                'selected_geometry in each floor audit is an uncalibrated heuristic, not deployed connectivity.',
                'No R02/R03 overall pass, calibration or formal-training gate is established.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-job', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu'); parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.pipeline_job, args.output, args.device, args.poll_seconds, args.resume), indent=2))


if __name__ == '__main__': main()
