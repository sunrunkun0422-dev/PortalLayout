#!/usr/bin/env python3
"""Freeze an RGB-only development cohort and audit historical evidence.

This command never trains and never accesses official val/test examples. It
does not infer that pre-trained weights are independent of the new cohort.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portallayout.datasets.zind_floor_dataset import ZInDFloorDataset, safe_path
from portallayout.datasets.floor_auxiliary_supervision import house_protocol
from portallayout.utils.room_feature_cache import fingerprint, write_json_atomic


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def choose_cohort(samples, protocol, excluded_houses, seed=123):
    """Sample without replacement, respecting historical fit/dev grouping."""
    eligible = collections.defaultdict(list)
    for index, row in enumerate(samples):
        if row['official_split'] == 'train' and 2 <= row['observation_count'] <= 80:
            eligible[row['house_id']].append((index, row))
    used = set(excluded_houses)
    chosen = {}
    role_pools = [('fit', 'fit', 40), ('selection', 'selection', 10),
                  ('calibration', 'calibration', 10)]
    for role, source_role, count in role_pools:
        pool = set(protocol[source_role]) & set(eligible) - used
        houses = sorted(pool, key=lambda h: (fingerprint(['design10', seed, role, h]), h))[:count]
        if len(houses) != count:
            raise ValueError('insufficient eligible houses for ' + role)
        chosen[role] = houses
        used.update(houses)
    pool = (set(protocol['selection']) | set(protocol['calibration'])) & set(eligible) - used
    houses = sorted(pool, key=lambda h: (fingerprint(['design10', seed, 'audit', h]), h))[:10]
    if len(houses) != 10:
        raise ValueError('insufficient eligible houses for audit')
    chosen['audit'] = houses
    rows = []
    for role, houses in chosen.items():
        for house in houses:
            index, sample = min(eligible[house], key=lambda v: (
                fingerprint(['design10', seed, house, v[1]['floor_uid']]), v[1]['floor_uid']))
            rows.append({'role': role, 'house_id': house, 'floor_uid': sample['floor_uid'],
                         'observations': sample['observation_count'],
                         'input_manifest': sample['input_manifest'], 'dataset_index': index})
    return chosen, rows


def recompute_geometry(path):
    grouped = collections.defaultdict(list)
    with open(path, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            grouped[row['collection']].append(row)
    result = {}
    for group, rows in grouped.items():
        if len({row['pair_id'] for row in rows}) != len(rows):
            raise ValueError('duplicate pair in geometry collection')
        result[group] = {'denominator': len(rows),
                         'dominant_wall_s1p0_success': sum(int(r['dominant_wall_s1p0']) for r in rows),
                         'local_m2_s0p0_success': sum(int(r['local_m2_s0p0']) for r in rows),
                         'houses': len({r['house_id'] for r in rows})}
    return result


def verify_cache(cache_root):
    contract = json.loads((cache_root / 'cache_contract.json').read_text())
    records = list((cache_root / 'observations').glob('*/*.json'))
    errors = []
    for path in records:
        try:
            row = json.loads(path.read_text())
            tensor = safe_path(cache_root, row['tensor_file'])
            if (row['contract_fingerprint'] != contract['fingerprint'] or
                    row['observation_id'] != path.parent.name or
                    not tensor.is_file() or sha256(tensor) != row['tensor_sha256']):
                errors.append(str(path))
        except (OSError, KeyError, ValueError) as error:
            errors.append('{}: {}'.format(path, error))
    run = json.loads((cache_root / 'run_report.json').read_text())
    return {'fingerprint': contract['fingerprint'], 'expected_unique': run['unique_observations'],
            'metadata_and_tensor_checked': len(records), 'errors': errors,
            'ok': len(records) == run['unique_observations'] and not errors,
            'historic_code_contract': 'pre-refactor; do not force resume under current identity',
            'contract_file_sha256': sha256(cache_root / 'cache_contract.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/ZInD-FloorSet-v1/train_20261021')
    p.add_argument('--raw-root', type=Path, default=ROOT.parent / 'zind/data')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    dataset_root, raw_root, output = args.dataset.resolve(), args.raw_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be absent or empty; never overwrite frozen results')
    historic = ROOT / 'experiments/research1_main_20261024/cohort/cohort.json'
    old_cohort = json.loads(historic.read_text())['cohort']
    excluded = {row['house_id'] for row in old_cohort}
    dataset = ZInDFloorDataset(dataset_root, official_split='train')
    protocol = house_protocol(dataset_root, args.seed)
    roles, cohort = choose_cohort(dataset.samples, protocol, excluded, args.seed)
    partition = json.loads((dataset_root / 'splits/official_house_split.json').read_text())
    assert not (set(partition['train']) & (set(partition['val']) | set(partition['test'])))
    assert not (set(partition['val']) & set(partition['test']))
    assert set().union(*(set(v) for v in roles.values())) <= set(partition['train'])
    assert sum(map(len, roles.values())) == len(set().union(*(set(v) for v in roles.values())))
    output.mkdir(parents=True, exist_ok=True)
    missing_rgb = []
    input_hashes = {}
    for row in cohort:
        manifest = dataset[row['dataset_index']]  # Strict RGB-only schema, no GT.
        if manifest['floor_uid'] != row['floor_uid']:
            raise ValueError('dataset index and frozen floor differ')
        path = safe_path(dataset_root, row['input_manifest'])
        input_hashes[row['floor_uid']] = sha256(path)
        write_json_atomic(output / 'manifests' / (row['floor_uid'] + '.json'), manifest)
        for obs in manifest['observations']:
            source = safe_path(raw_root, obs['rgb']['relative_path'])
            if not source.is_file():
                missing_rgb.append(str(source))
    cohort_report = {'schema_version': 'research10.cohort.v1', 'seed': args.seed,
                     'dataset_root': str(dataset_root), 'raw_root': str(raw_root),
                     'roles': roles, 'cohort': cohort, 'input_manifest_sha256': input_hashes,
                     'selection_uses': ['official_train_membership', 'house_id', 'observation_count', 'floor_uid'],
                     'historical_60_houses_excluded': sorted(excluded), 'missing_rgb': missing_rgb,
                     'warning': 'audit is in historical selection/calibration pools: exclude it from all supervision; '
                                'old pretrained model exposure is unknown.'}
    write_json_atomic(output / 'cohort.json', cohort_report)
    geometry = recompute_geometry(ROOT / 'docs/method_advantage_evidence_20261028/geometry_paired_selection.csv')
    metric = json.loads((ROOT / 'experiments/zind_observable_v1/results/best_metrics.json').read_text())['validation']['predicted_openings']
    review = json.loads((ROOT / 'experiments/research1_round2_20261024/review/review_template.json').read_text())
    review_counts = collections.Counter(row['judgement'] or 'pending' for row in review['cases'])
    calibration = json.loads((ROOT / 'experiments/research1_round2_20261024/R3_final_matching.json').read_text())['calibration']
    weights = {
        'frontend_layout': ROOT / 'checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl',
        'frontend_opening': ROOT / 'checkpoints/Opening_Head/zind_matching_v1/best.pt',
        'historical_matcher_r3': ROOT / 'experiments/research1_round2_20261024/R3/checkpoint.pt',
        'salve_verifier': ROOT.parent / 'salve/checkpoints/salve/1200ffbe47d836557d88fef052952337.pth',
    }
    exposure = {name: {'path': str(path), 'exists': path.is_file(),
                       'sha256': sha256(path) if path.is_file() else None,
                       'end_to_end_unseen_house_status': 'unknown_unverified'} for name, path in weights.items()}
    known = json.loads((ROOT / 'experiments/zind_observable_v1/training/selection_summary.json').read_text())
    exposure['observable_joint_training_house_counts'] = {
        'fit': len(known['train_houses']), 'internal_monitor': len(known['internal_dev_houses']),
        'note': 'applies to joint observable experiment, not proof of frontend checkpoint origin'}
    cache = verify_cache(ROOT / 'outputs/research1_floor_frontend/main60_20261024')
    report = {'schema_version': 'research10.r01.audit.v1',
              'created_at': datetime.now().astimezone().isoformat(),
              'status': 'diagnostic_only_exposure_unknown' if cache['ok'] and not missing_rgb else 'failed_integrity',
              'role_counts': {k: len(v) for k, v in roles.items()},
              'official_house_counts': {k: len(v) for k, v in partition.items()},
              'old_cohort_excluded': len(excluded), 'missing_rgb_count': len(missing_rgb),
              'exposure': exposure, 'historical_cache': cache,
              'geometry_recomputed': geometry,
              'observable_validation': {k: metric[k] for k in ('positive_pair_count', 'negative_pair_count',
                  'positive_candidate_recall', 'conditional_candidate_top1', 'end_to_end_candidate_top1',
                  'negative_rejection_rate')},
              'observable_validation_scope': 'internal train-house monitor, not official test',
              'manual_review_counts': dict(review_counts), 'historical_opening_calibration': calibration,
              'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in [
                  Path(__file__).resolve(), historic,
                  ROOT / 'docs/method_advantage_evidence_20261028/geometry_paired_selection.csv',
                  ROOT / 'experiments/zind_observable_v1/results/best_metrics.json',
                  ROOT / 'experiments/research1_round2_20261024/review/review_template.json',
                  ROOT / 'experiments/research1_round2_20261024/R3_final_matching.json']},
              'dataset_source_sha256': {name: sha256(dataset_root / name) for name in (
                  'samples/all_panoramas.jsonl', 'splits/official_house_split.json', 'splits/derived_house_split.json')},
              'limitations': ['no claim of unseen-house end-to-end generalization',
                  'historic development data reused for historical diagnostic only',
                  'new cohort has not been encoded or trained', 'manual labels remain pending']}
    write_json_atomic(output / 'audit.json', report)
    print(json.dumps({'status': report['status'], 'roles': report['role_counts'],
                      'cache_records_checked': cache['metadata_and_tensor_checked'],
                      'cache_errors': len(cache['errors']), 'missing_rgb': len(missing_rgb),
                      'geometry_recomputed': geometry}, ensure_ascii=False, indent=2))
    return 0 if cache['ok'] and not missing_rgb else 2


if __name__ == '__main__':
    raise SystemExit(main())
