#!/usr/bin/env python3
"""Offline correspondence diagnostics; never used to supply inference inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portallayout.datasets.floor_auxiliary_supervision import house_role
from portallayout.evaluation.floor_metrics import load_floor_ground_truth
from portallayout.engine.floor_calibration import opening_workpoint
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def summarize(rows, threshold):
    """Recall includes unretrieved positives; unknown never becomes a negative."""
    matches = [m for row in rows for m in row['matches'] if m['confidence'] >= threshold]
    tp = sum(m['target'] == 1 for m in matches)
    fp = sum(m['target'] == 0 for m in matches)
    positives = sum(r['positive_cells'] for r in rows)
    negative_pairs = [r for r in rows if r['known_negative_pair']]
    bad = sum(any(m['confidence'] >= threshold for m in r['matches']) for r in negative_pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    return dict(pairs=len(rows), houses=len({r['house_id'] for r in rows}), threshold=threshold,
                positive_cells=positives, accepted_true=tp, accepted_false=fp,
                accepted_unknown=sum(m['target'] < 0 for m in matches),
                precision_known=precision, recall_all_weak_positives=recall,
                f1_known=2 * tp / (positives + tp + fp) if positives + tp + fp else None,
                known_negative_pairs=len(negative_pairs), false_accepted_negative_pairs=bad,
                negative_pair_false_acceptance=bad / len(negative_pairs) if negative_pairs else None,
                numerical_status=dict(Counter(r['numerical_status'] for r in rows)),
                unretrieved_positive_cells=positives-sum(m['target'] == 1 for r in rows for m in r['matches']))


def join_predictions(weak, auxiliary, ground_truth):
    if weak['source_snapshot'] != auxiliary['source_snapshot']:
        raise ValueError('prediction and weak-label cache snapshots differ')
    predictions = {r['pair_id']: r for r in auxiliary['opening_pairs']}
    if len(predictions) != len(auxiliary['opening_pairs']):
        raise ValueError('duplicate predicted pair identity')
    rows = []
    used = set()
    for label in weak['pairs']:
        aid, bid = label['observation_a'], label['observation_b']
        # The dataset and prediction schemas intentionally use different pair IDs.
        key = fingerprint([aid, bid])
        predicted = predictions.get(key)
        used.add(key)
        role = house_role(label['house_id'], auxiliary['house_protocol'])
        if predicted and (predicted['house_id'] != label['house_id'] or predicted['role'] != role):
            raise ValueError('predicted pair house or role mismatch')
        cells = label['weak_cells']
        matches, occupied_a, occupied_b = [], set(), set()
        for match in predicted['matches'] if predicted else []:
            i, j = match['index_a'], match['index_b']
            if not (0 <= i < len(label['candidate_ids_a']) and 0 <= j < len(label['candidate_ids_b'])):
                raise ValueError('match candidate index outside source labels')
            if (match['candidate_id_a'] != label['candidate_ids_a'][i] or
                    match['candidate_id_b'] != label['candidate_ids_b'][j]):
                raise ValueError('match candidate identity mismatch')
            if i in occupied_a or j in occupied_b:
                raise ValueError('matches must be one-to-one')
            occupied_a.add(i); occupied_b.add(j)
            if not 0 <= match['confidence'] <= 1:
                raise ValueError('invalid match confidence')
            if match['target'] != cells[i][j]:
                raise ValueError('exported target differs from original weak label')
            matches.append(dict(confidence=match['confidence'], target=cells[i][j]))
        ga, gb = (ground_truth[label['floor_uid']]['observations'][uid] for uid in (aid, bid))
        relationship = ('same_room' if ga['room_id'] == gb['room_id'] else 'cross_room') if (
            ga['membership_confirmed'] and gb['membership_confirmed']) else 'unknown_room'
        flat = [v for row in cells for v in row]
        rows.append(dict(pair_id=key, floor_uid=label['floor_uid'], house_id=label['house_id'], role=role,
                         relationship=relationship, positive_cells=flat.count(1),
                         known_negative_pair=bool(flat and all(v == 0 for v in flat)), matches=matches,
                         numerical_status=predicted.get('numerical_status') if predicted else 'missing_prediction'))
    if set(predictions)-used:
        raise ValueError('prediction pairs absent from source supervision')
    return rows


def checked_json(path):
    data = json.loads(Path(path).read_text())
    claimed = data.pop('fingerprint')
    if fingerprint(data) != claimed:
        raise ValueError('invalid source fingerprint: '+str(path))
    data['fingerprint'] = claimed
    return data


def frozen_audit_threshold(path, protocol, checkpoint_path):
    frozen = checked_json(path)
    calibration = frozen.get('calibration', {})
    if (frozen.get('house_protocol') != protocol or
            frozen.get('pair_checkpoint_sha256') != content_hash(checkpoint_path) or
            calibration.get('status') != 'calibrated' or
            'calibration' not in frozen.get('roles', {}) or 'audit' in frozen.get('roles', {})):
        raise ValueError('frozen threshold report does not match protocol, checkpoint, or calibration status')
    threshold = calibration.get('threshold')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError('invalid frozen calibration threshold')
    return calibration, threshold


def lightweight_report(args):
    import torch
    from portallayout.datasets.floor_auxiliary_supervision import (house_protocol, validate_cache_house_roles,
                                                                  validate_supervision_house_roles)
    from portallayout.evaluation.opening_matching import evaluate_matching, room_scopes, summary, quality_metrics
    from portallayout.engine.floor_calibration import strict_opening_workpoint
    from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
    from portallayout.engine.opening_pair_training import read_supervision
    from portallayout.utils.opening_pair_checkpoint import load_checkpoint
    torch.set_num_threads(2)
    cache = FrozenOpeningCache(args.prediction_cache)
    audit_only = 'audit' in args.roles
    weak = read_supervision(args.weak_supervision, cache, allow_audit=audit_only)
    encoder, matcher, checkpoint = load_checkpoint(args.pair_checkpoint, args.device)
    if checkpoint.get('training_contract', {}).get('source_contract', cache.fingerprint) != cache.fingerprint:
        raise ValueError('checkpoint frontend contract differs from prediction cache')
    protocol = house_protocol(args.dataset, args.protocol_seed, args.house_protocol)
    if checkpoint.get('training_contract', {}).get('house_protocol', protocol) != protocol:
        raise ValueError('checkpoint and report house protocols differ')
    if audit_only:
        if not args.house_protocol or args.roles != ['audit'] or args.calibrate or not args.threshold_report:
            raise ValueError('audit requires only audit role, custom protocol, frozen threshold report, and no calibration')
        if weak.get('usage') != 'audit_only' or weak.get('house_protocol') != protocol:
            raise ValueError('audit requires dedicated audit-only weak supervision bound to the same protocol')
        contract = checkpoint.get('training_contract', {})
        if (contract.get('house_protocol') != protocol or contract.get('source_contract') != cache.fingerprint or
                checkpoint.get('training_status') != 'trained'):
            raise ValueError('audit checkpoint must be fully trained and explicitly bound to this protocol and frontend')
        floors = validate_cache_house_roles(cache, args.dataset, protocol, {'audit'},
                                           Path(args.house_protocol).resolve().parent/'cohort.json')
        validate_supervision_house_roles(weak, protocol, {'audit'}, floors, cache)
        calibration, threshold = frozen_audit_threshold(args.threshold_report, protocol, args.pair_checkpoint)
    else:
        if weak.get('usage') == 'audit_only': raise ValueError('audit supervision requires audit-only reporting')
        if args.house_protocol:
            if weak.get('house_protocol') != protocol:
                raise ValueError('weak supervision and report house protocols differ')
            floors = validate_cache_house_roles(cache, args.dataset, protocol, {'fit', 'selection', 'calibration'},
                                               Path(args.house_protocol).resolve().parent/'cohort.json')
            validate_supervision_house_roles(weak, protocol, {'fit', 'selection', 'calibration'}, floors, cache)
        calibration, threshold = None, None
    if args.calibrate and ('calibration' not in args.roles or checkpoint['training_status'] != 'trained'):
        raise ValueError('calibration requires full-trained checkpoint and calibration role')
    source_rows = [r for r in weak['pairs'] if house_role(r['house_id'], protocol) in args.roles]
    scopes, annotation_hashes = room_scopes(source_rows, args.dataset)
    contract = checkpoint.get('training_contract', {})
    if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    records, runtime = evaluate_matching(cache, encoder, matcher, source_rows, scopes, args.device,
        contract.get('objective', 'assignment'), contract.get('auxiliary_weight', .5), contract.get('margin', .2),
        output_file=Path(args.output).with_suffix('.pairs.jsonl'))
    for r in records: r['role'] = house_role(r['house_id'], protocol)
    if not audit_only:
        calibration = strict_opening_workpoint(records) if args.calibrate else {'status': 'not_requested'}
        threshold = calibration.get('threshold')
    def tables(rows):
        return dict(diagnostic_zero_threshold=summary(rows),
                    calibrated=summary(rows, threshold) if threshold is not None else None)
    result = dict(schema_version='opening-matching-diagnostics.v2', label_source='geometry_weak',
                  source_snapshot=cache.snapshot, source_contract=cache.fingerprint,
                  pair_checkpoint_sha256=content_hash(args.pair_checkpoint), house_protocol=protocol,
                  weak_supervision_sha256=content_hash(args.weak_supervision),
                  pair_results_sha256=content_hash(Path(args.output).with_suffix('.pairs.jsonl')),
                  annotation_sha256=annotation_hashes, calibration=calibration, roles={},
                  candidate_associations=weak['statistics']['association_status'], runtime=runtime)
    if audit_only:
        result['frozen_threshold_report_sha256'] = content_hash(args.threshold_report)
        result['audit_note'] = 'Frozen calibration threshold applied without selection or retuning on audit houses.'
    for role in args.roles:
        subset = [r for r in records if r['role'] == role]
        result['roles'][role] = dict(tables(subset), quality=quality_metrics(subset),
            relationships={scope: tables([r for r in subset if r['relationship'] == scope])
                           for scope in ('same_room', 'cross_room', 'unknown_room')},
            floors={uid: tables([r for r in subset if r['floor_uid'] == uid]) for uid in sorted({r['floor_uid'] for r in subset})})
    subset = [r for r in records if r['role'] == 'selection']
    result['selection_pr'] = [summary(subset, t) for t in sorted({0., 1.} |
                              {m['confidence'] for r in subset for m in r['matches'] if m['target'] >= 0})]
    if any(content_hash(p) != h for p, h in annotation_hashes.items()): raise ValueError('annotations changed')
    result['source_files_verified'] = cache.check_unchanged()
    result['peak_allocated_MiB'] = torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None
    result['fingerprint'] = fingerprint(result)
    write_json_atomic(args.output, result)
    print(json.dumps({'calibration': calibration, 'pairs': len(records),
                      'roles': {k: v['quality'] for k, v in result['roles'].items()}}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weak-supervision', required=True)
    parser.add_argument('--auxiliary-supervision')
    parser.add_argument('--prediction-cache'); parser.add_argument('--pair-checkpoint')
    parser.add_argument('--protocol-seed', type=int, default=123)
    parser.add_argument('--house-protocol'); parser.add_argument('--threshold-report')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--roles', nargs='+', choices=('fit','selection','calibration','audit'), default=['fit','selection'])
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('output must be new')
    if args.prediction_cache or args.pair_checkpoint:
        if not args.prediction_cache or not args.pair_checkpoint or args.auxiliary_supervision:
            parser.error('lightweight mode needs cache/checkpoint and no auxiliary supervision')
        return lightweight_report(args)
    if args.house_protocol or args.threshold_report or 'audit' in args.roles:
        parser.error('custom-protocol audit reporting requires lightweight cache/checkpoint mode')
    if not args.auxiliary_supervision: parser.error('provide auxiliary supervision or prediction cache/checkpoint')
    weak, auxiliary = checked_json(args.weak_supervision), checked_json(args.auxiliary_supervision)
    gt = {uid: load_floor_ground_truth(args.dataset, uid) for uid in sorted({r['floor_uid'] for r in weak['pairs']})}
    rows = join_predictions(weak, auxiliary, gt)
    calibration = opening_workpoint(auxiliary['opening_pairs'])
    threshold = calibration.get('threshold')
    report = dict(schema_version='opening-matching-diagnostics.v1', label_source='geometry_weak',
                  source_snapshot=weak['source_snapshot'], pair_checkpoint_sha256=auxiliary['pair_checkpoint_sha256'],
                  input_sha256={p: content_hash(p) for p in (args.weak_supervision, args.auxiliary_supervision)},
                  annotation_sha256={p: content_hash(p) for floor in gt.values() for p in floor['source_files']},
                  calibration=calibration, candidate_associations=weak['statistics']['association_status'], roles={})
    for role in ('fit', 'selection', 'calibration'):
        subset = [r for r in rows if r['role'] == role]
        report['roles'][role] = dict(
            diagnostic_zero_threshold=summarize(subset, 0.),
            calibrated=summarize(subset, threshold) if threshold is not None else None,
            relationships={name: dict(diagnostic_zero_threshold=summarize([r for r in subset if r['relationship'] == name], 0.),
                                     calibrated=summarize([r for r in subset if r['relationship'] == name], threshold) if threshold is not None else None)
                           for name in ('same_room', 'cross_room', 'unknown_room')},
            floors={uid: summarize([r for r in subset if r['floor_uid'] == uid],
                                   threshold if threshold is not None else 0.)
                    for uid in sorted({r['floor_uid'] for r in subset})})
    # Selection curves diagnose failure without tuning against calibration houses.
    subset = [r for r in rows if r['role'] == 'selection']
    scores = sorted({0., 1.} | {m['confidence'] for r in subset for m in r['matches']})
    report['selection_pr'] = [summarize(subset, t) for t in scores]
    report['notes'] = ['Recall denominator includes all weak-positive candidate cells, including missing predictions.',
                       'Room membership is read only for offline stratification.',
                       'Zero-threshold output is diagnostic; it is not an accepted pose graph.',
                       'Candidate association counts are not annotated-opening recall; manual-review metrics are separate.']
    write_json_atomic(output, report)
    print(json.dumps({'calibration': calibration, 'roles': {k: v['diagnostic_zero_threshold']
                                                          for k, v in report['roles'].items()}}, indent=2))


if __name__ == '__main__':
    main()
