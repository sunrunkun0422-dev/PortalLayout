#!/usr/bin/env python3
"""Export fixed-candidate R04 baseline scores from an existing prediction cache.

The two phases are explicit: `plan` freezes observation/candidate IDs and
`score` verifies and scores that unchanged plan using a specified pair
checkpoint. Only official-train fit/selection outputs may be scored. This tool
never reads labels, weak supervision, GT poses, or GT camera heights.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from portallayout.datasets.floor_auxiliary_supervision import house_protocol, validate_cache_house_roles
from portallayout.datasets.zind_floor_dataset import safe_path
from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
from portallayout.utils.opening_pair_checkpoint import load_checkpoint
from portallayout.utils.opening_representation_adapter import prepare_opening_representation_batch
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic
from tools.research10_matching_baselines import score_fixed_candidates


PAIR_KEYS = {'pair_id', 'floor_uid', 'house_id', 'role', 'observation_a', 'observation_b',
             'candidate_ids_a', 'candidate_ids_b'}
PLAN_KEYS = {'schema', 'source_contract', 'source_snapshot', 'cohort_sha256',
             'house_protocol_fingerprint', 'roles', 'seed', 'max_pairs', 'population_pairs',
             'population_by_role', 'selected_pairs', 'pairs', 'fingerprint'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_context(cache_root, dataset_root, protocol_path):
    """Validate cache, official-train membership and exact frozen development cohort."""
    cache_root, dataset_root, protocol_path = map(lambda p: Path(p).resolve(),
                                                  (cache_root, dataset_root, protocol_path))
    cohort_path = protocol_path.parent / 'cohort.json'
    protocol = house_protocol(dataset_root, protocol_path=protocol_path)
    cohort = read_json(cohort_path)
    if Path(cohort['dataset_root']).resolve() != dataset_root:
        raise ValueError('dataset root does not match cohort')
    rows = cohort.get('cohort', [])
    row_by_floor = {r['floor_uid']: r for r in rows}
    if not rows or len(row_by_floor) != len(rows):
        raise ValueError('cohort floor identities invalid')
    groups = {role: set(protocol[role]) for role in ('fit', 'selection', 'calibration', 'audit')}
    if any(r['role'] not in groups or r['house_id'] not in groups[r['role']] for r in rows):
        raise ValueError('cohort role and house mismatch')
    marker = read_json(cache_root / 'cache_contract.json')
    contract = marker['contract']
    if contract.get('uses_ground_truth') is not False or contract.get('cohort_sha256') != content_hash(cohort_path):
        raise ValueError('cache must use the frozen prediction-only cohort contract')
    expected = {uid for uid, r in row_by_floor.items() if r['role'] in ('fit', 'selection', 'calibration')}
    for floor_uid in expected:
        row = row_by_floor[floor_uid]
        if content_hash(safe_path(dataset_root, row['input_manifest'])) != cohort['input_manifest_sha256'][floor_uid]:
            raise ValueError('cohort RGB input manifest hash mismatch')
    cache = FrozenOpeningCache(cache_root)
    floor_houses = validate_cache_house_roles(cache, dataset_root, protocol,
                         {'fit', 'selection', 'calibration'}, cohort_path=cohort_path)
    if set(floor_houses) != expected:
        raise ValueError('cache does not exactly cover development cohort')
    return {'cache': cache, 'dataset_root': dataset_root, 'protocol': protocol,
            'protocol_path': protocol_path, 'cohort_path': cohort_path,
            'cohort': cohort, 'row_by_floor': row_by_floor}


def candidate_ids(cache, observation_id):
    record = cache.records.get(observation_id)
    if record is None:
        return None  # Prediction failure, not an empty candidate set.
    candidates = record['prediction']['openings']
    ids = [c['candidate_id'] for c in candidates]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('invalid or duplicate cached candidate ID')
    return ids


def make_plan(context, roles, max_pairs, seed=123):
    if not roles or not set(roles) <= {'fit', 'selection'}:
        raise ValueError('only fit and selection roles can be scored')
    if max_pairs < 0:
        raise ValueError('max_pairs must be nonnegative; 0 means all')
    cache = context['cache']
    population, seen = [], set()
    for floor in cache.floors:
        meta = context['row_by_floor'][floor['floor_uid']]
        if meta['role'] not in roles:
            continue
        ids = sorted(floor['observation_order'])
        for a, b in itertools.combinations(ids, 2):
            pair_id = fingerprint([a, b])
            if pair_id in seen:
                raise ValueError('duplicate observation pair identity')
            seen.add(pair_id)
            population.append({'pair_id': pair_id, 'floor_uid': floor['floor_uid'],
                               'house_id': meta['house_id'], 'role': meta['role'],
                               'observation_a': a, 'observation_b': b,
                               'candidate_ids_a': candidate_ids(cache, a),
                               'candidate_ids_b': candidate_ids(cache, b)})
    population.sort(key=lambda p: (fingerprint([seed, p['pair_id']]), p['pair_id']))
    selected = population if max_pairs == 0 else population[:max_pairs]
    result = {'schema': 'r04-cache-fixed-pairs.v1', 'source_contract': cache.fingerprint,
              'source_snapshot': cache.snapshot, 'cohort_sha256': content_hash(context['cohort_path']),
              'house_protocol_fingerprint': context['protocol']['fingerprint'],
              'roles': sorted(roles), 'seed': seed, 'max_pairs': max_pairs,
              'population_pairs': len(population),
              'population_by_role': {r: sum(p['role'] == r for p in population) for r in sorted(roles)},
              'selected_pairs': len(selected), 'pairs': selected}
    result['fingerprint'] = fingerprint(result)
    cache.check_unchanged()
    return result


def validate_plan(plan, context):
    if not isinstance(plan, dict) or set(plan) != PLAN_KEYS or plan.get('schema') != 'r04-cache-fixed-pairs.v1':
        raise ValueError('unsupported fixed-pair plan')
    if fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'}) != plan.get('fingerprint'):
        raise ValueError('fixed-pair plan fingerprint mismatch')
    cache = context['cache']
    if (plan.get('source_contract') != cache.fingerprint or plan.get('source_snapshot') != cache.snapshot or
            plan.get('cohort_sha256') != content_hash(context['cohort_path']) or
            plan.get('house_protocol_fingerprint') != context['protocol']['fingerprint']):
        raise ValueError('fixed-pair plan source contract mismatch')
    roles = plan.get('roles')
    if not isinstance(roles, list) or not roles or not set(roles) <= {'fit', 'selection'}:
        raise ValueError('fixed-pair plan has forbidden roles')
    floors = {f['floor_uid']: f for f in cache.floors}
    rows = plan.get('pairs')
    if not isinstance(rows, list) or plan.get('selected_pairs') != len(rows):
        raise ValueError('fixed-pair plan count mismatch')
    seen = set()
    for p in rows:
        if not isinstance(p, dict) or set(p) != PAIR_KEYS:
            raise ValueError('fixed-pair plan permits only predicted IDs and cohort identity')
        uid, role = p['floor_uid'], p['role']
        meta, floor = context['row_by_floor'].get(uid), floors.get(uid)
        if not meta or not floor or role not in roles or role not in ('fit', 'selection'):
            raise ValueError('pair floor or role is not permitted')
        if meta['role'] != role or meta['house_id'] != p['house_id']:
            raise ValueError('pair house or role differs from cohort')
        a, b = p['observation_a'], p['observation_b']
        if a >= b or a not in floor['observation_order'] or b not in floor['observation_order']:
            raise ValueError('invalid observation pair')
        if p['pair_id'] != fingerprint([a, b]) or p['pair_id'] in seen:
            raise ValueError('duplicate or invalid observation pair ID')
        seen.add(p['pair_id'])
        if p['candidate_ids_a'] != candidate_ids(cache, a) or p['candidate_ids_b'] != candidate_ids(cache, b):
            raise ValueError('fixed candidate identities or ordering changed')
    return rows


def load_bound_encoder(context, checkpoint, device):
    encoder, _, state = load_checkpoint(checkpoint, device=device)
    contract = state.get('training_contract') or {}
    cache = context['cache']
    if (contract.get('source_contract') != cache.fingerprint or
            contract.get('source_snapshot') != cache.snapshot or
            contract.get('house_protocol') != context['protocol']):
        raise ValueError('checkpoint source contract, snapshot or house protocol mismatch')
    if not set(contract.get('fit_houses', ())) <= set(context['protocol']['fit']):
        raise ValueError('checkpoint training house role mismatch')
    if state.get('training_status') not in ('smoke_trained', 'trained'):
        raise ValueError('checkpoint has no completed train/smoke status')
    encoder.eval()
    return encoder, state


def encode_view(cache, encoder, uid, device, scale_assumption):
    record, arrays = cache.load(uid)
    prepared = prepare_opening_representation_batch([record], [arrays], device)
    with torch.no_grad():
        encoded = encoder(*prepared.encoder_inputs())
    ids = list(prepared.candidates.candidate_ids[0])
    if ids != candidate_ids(cache, uid) or list(encoded.candidate_ids[0]) != ids:
        raise ValueError('encoded candidate IDs differ from frozen candidate IDs')
    embeddings = encoded.baseline_descriptors[0].detach().cpu().numpy()
    valid = encoded.valid_mask[0].detach().cpu().numpy()
    widths = prepared.predicted_geometry.wall_attributes[0, :, 0].detach().cpu().numpy()
    width_mask = prepared.predicted_geometry.wall_attribute_valid[0, :, 0].detach().cpu().numpy()
    candidates = []
    for i, uid_candidate in enumerate(ids):
        vector = np.asarray(embeddings[i], dtype=float)
        finite = np.isfinite(vector)
        width_valid = bool(width_mask[i]) and bool(np.isfinite(widths[i]))
        candidates.append({'candidate_id': uid_candidate, 'valid': bool(valid[i]),
                           'appearance': np.where(finite, vector, 0.).tolist(),
                           'appearance_valid': finite.tolist(),
                           'opening_width': float(widths[i]) if width_valid else None,
                           'width_valid': width_valid})
    scale = ({'valid': True, 'factor_to_common': 1., 'common_unit': 'shared_camera_height',
              'source': 'shared_height_assumption'} if scale_assumption else
             {'valid': False, 'factor_to_common': None, 'common_unit': None, 'source': 'unavailable'})
    return {'view_id': uid, 'appearance_source': 'predicted_single_view',
            'width_source': 'predicted_single_view', 'scale': scale, 'candidates': candidates}


def score_plan(context, plan, encoder, device='cpu', scale_assumption=False):
    rows = validate_plan(plan, context)
    cache = context['cache']
    views, results = {}, []
    for p in rows:
        row = {k: p[k] for k in ('pair_id', 'floor_uid', 'house_id', 'role', 'observation_a', 'observation_b')}
        if p['candidate_ids_a'] is None or p['candidate_ids_b'] is None:
            row.update(status='prediction_unavailable', candidate_pair_count=0, scores=[])
        else:
            for side in ('a', 'b'):
                uid = p['observation_' + side]
                if uid not in views:
                    views[uid] = encode_view(cache, encoder, uid, device, scale_assumption)
            fixed = [{'candidate_id_a': a, 'candidate_id_b': b}
                     for a in p['candidate_ids_a'] for b in p['candidate_ids_b']]
            scored = score_fixed_candidates({'schema': 'r04-fixed-candidates.v1',
                                              'view_a': views[p['observation_a']],
                                              'view_b': views[p['observation_b']], 'candidate_pairs': fixed})
            row.update(status='scored' if fixed else 'no_candidate_pairs',
                       candidate_pair_count=scored['pair_count'], scores=scored['rows'])
        results.append(row)
    cache.check_unchanged()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('plan', 'score'):
        q = sub.add_parser(cmd)
        q.add_argument('--cache', required=True); q.add_argument('--dataset', required=True)
        q.add_argument('--house-protocol', required=True); q.add_argument('--output', required=True)
        if cmd == 'plan':
            q.add_argument('--roles', choices=('fit', 'selection', 'both'), default='both')
            q.add_argument('--max-pairs', type=int, required=True, help='0 = all observation pairs')
            q.add_argument('--seed', type=int, default=123)
        else:
            q.add_argument('--plan', required=True); q.add_argument('--checkpoint', required=True)
            q.add_argument('--device', default='cpu')
            q.add_argument('--shared-camera-height-assumption', action='store_true')
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        p.error('output already exists; choose a new file')
    context = load_context(args.cache, args.dataset, args.house_protocol)
    if args.command == 'plan':
        roles = ('fit', 'selection') if args.roles == 'both' else (args.roles,)
        result = make_plan(context, roles, args.max_pairs, args.seed)
        write_json_atomic(output, result)
        print(json.dumps({'population_pairs': result['population_pairs'], 'selected_pairs': result['selected_pairs']}))
        return
    plan = read_json(args.plan)
    validate_plan(plan, context)
    encoder, state = load_bound_encoder(context, args.checkpoint, args.device)
    rows = score_plan(context, plan, encoder, args.device, args.shared_camera_height_assumption)
    result = {'schema': 'r04-cache-baseline-export.v1', 'source_contract': context['cache'].fingerprint,
              'source_snapshot': context['cache'].snapshot, 'fixed_plan_sha256': content_hash(args.plan),
              'checkpoint_sha256': content_hash(args.checkpoint), 'checkpoint_weights_hash': state['weights_hash'],
              'checkpoint_training_status': state['training_status'], 'checkpoint_completion_independently_verified': False,
              'feature_source': 'encoder.baseline_descriptors from frozen cached predicted 1D features and opening probabilities; not pure RGB',
              'width_source': 'predicted_wall_attributes opening_width from predicted opening geometry',
              'runtime_source_sha256': {str(p.relative_to(ROOT)): content_hash(p) for p in (
                  Path(__file__).resolve(), ROOT/'tools/research10_matching_baselines.py',
                  ROOT/'portallayout/models/opening_representation.py',
                  ROOT/'portallayout/utils/opening_representation_adapter.py',
                  ROOT/'portallayout/utils/frozen_opening_cache.py')},
              'torch_version': torch.__version__, 'device': args.device,
              'geometry_scale': 'explicit_shared_camera_height_assumption' if args.shared_camera_height_assumption else 'unknown',
              'ground_truth_read': False, 'scoring_only': True, 'calibration': 'not_calibrated',
              'observation_pair_count': len(rows), 'observation_pairs': rows}
    write_json_atomic(output, result)
    print(json.dumps({'observation_pair_count': len(rows),
                      'candidate_pair_count': sum(r['candidate_pair_count'] for r in rows)}))


if __name__ == '__main__':
    main()
