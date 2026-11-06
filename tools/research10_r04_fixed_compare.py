#!/usr/bin/env python3
"""Offline fixed-candidate ranking comparison on development weak labels.

The learned matcher score, appearance descriptor cosine and predicted-width
ratio are compared on the same candidate cells. No threshold is tuned, and
ranking forcibly chooses one cell when a score exists; this is not a No-Match,
end-to-end accuracy, calibrated precision or held-out test evaluation.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def checked(path):
    data = json.loads(Path(path).read_text()); claimed = data.get('fingerprint')
    if claimed is None or fingerprint({k: v for k, v in data.items() if k != 'fingerprint'}) != claimed:
        raise ValueError('invalid fingerprint: '+str(path))
    return data


def labels_for_pair(row, a, b):
    """Align weak cells by observation identity, never by unrelated pair IDs."""
    ids_a, ids_b = row['candidate_ids_a'], row['candidate_ids_b']
    cells = np.asarray(row['weak_cells'], dtype=int).reshape(len(ids_a), len(ids_b))
    if not np.isin(cells, (-1, 0, 1)).all(): raise ValueError('invalid weak cell')
    if (row['observation_a'], row['observation_b']) == (a, b):
        return ids_a, ids_b, cells
    if (row['observation_a'], row['observation_b']) == (b, a):
        return ids_b, ids_a, cells.T
    raise ValueError('weak label observations differ')


def top1(items, cells):
    """An unlabeled selected cell stays unknown; missing scores remain missing."""
    usable = [(float(v), a, b) for a, b, v in items if v is not None and math.isfinite(float(v))]
    if not usable: return {'outcome': 'no_score', 'selected': None}
    value, a, b = sorted(usable, key=lambda x: (-x[0], x[1], x[2]))[0]
    if (a, b) not in cells: raise ValueError('scored candidate outside fixed weak cells')
    label = cells[(a, b)]
    return {'outcome': {1: 'correct', 0: 'incorrect', -1: 'unknown'}[label],
            'selected': [a, b], 'score': value}


def evaluate_pair(base, weak, raw, scope):
    a, b = base['observation_a'], base['observation_b']
    ids_a, ids_b, matrix = labels_for_pair(weak, a, b)
    if raw['observation_a'] != a or raw['observation_b'] != b:
        raise ValueError('matcher observation direction differs')
    if raw.get('candidate_ids_a') != ids_a or raw.get('candidate_ids_b') != ids_b:
        raise ValueError('matcher and weak-label candidate ordering differ')
    cells = {(x, y): int(matrix[i, j]) for i, x in enumerate(ids_a) for j, y in enumerate(ids_b)}
    seen = set(); appearance, width = [], []
    for entry in base['scores']:
        key = (entry['candidate_id_a'], entry['candidate_id_b'])
        if key in seen or key not in cells: raise ValueError('duplicate or foreign baseline candidate')
        seen.add(key)
        for name, target in (('appearance_cosine', appearance), ('width_ratio', width)):
            score = entry[name]
            target.append((*key, score['score'] if score['available'] else None))
    if base['status'] == 'scored' and seen != set(cells):
        raise ValueError('baseline does not cover all fixed candidate cells')
    learned = []
    if raw.get('numerical_status') == 'assigned':
        values = np.asarray(raw['scores'], dtype=float)
        if values.shape != matrix.shape or not np.isfinite(values).all():
            raise ValueError('matcher score matrix shape or values invalid')
        learned = [(x, y, values[i, j]) for i, x in enumerate(ids_a) for j, y in enumerate(ids_b)]
    return {'pair_id': base['pair_id'], 'floor_uid': base['floor_uid'], 'house_id': base['house_id'],
            'scope': scope, 'known_weak_positive_cells': int((matrix == 1).sum()),
            'known_weak_negative_cells': int((matrix == 0).sum()), 'unknown_cells': int((matrix == -1).sum()),
            'methods': {'matcher_raw_logit': top1(learned, cells),
                        'single_view_descriptor_cosine': top1(appearance, cells),
                        'opening_width_ratio': top1(width, cells)}}


def summarize(rows, scope='cross_room'):
    data = [r for r in rows if r['scope'] == scope]
    positives = [r for r in data if r['known_weak_positive_cells'] > 0]
    methods = ('matcher_raw_logit', 'single_view_descriptor_cosine', 'opening_width_ratio')
    out = {'scope': scope, 'pair_denominator': len(data), 'pairs_with_known_positive_candidate': len(positives),
           'pairs_without_known_positive_candidate': len(data)-len(positives), 'methods': {}}
    for method in methods:
        counts = Counter(r['methods'][method]['outcome'] for r in positives)
        by_house = defaultdict(list)
        for r in positives: by_house[r['house_id']].append(r['methods'][method]['outcome'] == 'correct')
        house_rates = {h: float(np.mean(values)) for h, values in sorted(by_house.items())}
        out['methods'][method] = {'known_positive_pair_outcomes': dict(counts),
            'correct_over_known_positive_pairs': counts['correct']/len(positives) if positives else None,
            'house_macro_correct_over_known_positive_pairs': float(np.mean(list(house_rates.values()))) if house_rates else None,
            'house_rates': house_rates, 'houses_with_known_positive_pairs': len(house_rates)}
    if positives:
        ref = out['methods']['matcher_raw_logit']['house_rates']
        for method in methods[1:]:
            base = out['methods'][method]['house_rates']
            if set(ref) != set(base): raise ValueError('house support mismatch')
            delta = np.asarray([ref[h]-base[h] for h in sorted(ref)])
            rng = np.random.RandomState(123)
            samples = delta[rng.randint(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
            out['methods'][method]['matcher_minus_baseline_house_macro_delta'] = float(delta.mean())
            out['methods'][method]['matcher_minus_baseline_house_bootstrap_ci95'] = np.percentile(samples, [2.5, 97.5]).tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True); p.add_argument('--fixed-plan', required=True)
    p.add_argument('--followup', required=True); p.add_argument('--weak-supervision', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args(); output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('use new output directory')
    baseline_path, plan_path, weak_path, followup = map(Path, (args.baseline, args.fixed_plan, args.weak_supervision, args.followup))
    baseline, plan, weak = json.loads(baseline_path.read_text()), checked(plan_path), checked(weak_path)
    contract, barrier, result = (checked(followup/n) for n in ('run_contract.json', 'prediction_barrier.json', 'summary.json'))
    if (result['status'] != 'diagnostic_only' or baseline.get('ground_truth_read') is not False or
        baseline.get('fixed_plan_sha256') != content_hash(plan_path) or
        baseline.get('checkpoint_sha256') != contract['checkpoint_sha256'] or
        baseline.get('source_snapshot') != contract['cache_snapshot'] or
        weak.get('source_snapshot') != contract['cache_snapshot'] or
        plan['roles'] != ['selection'] or plan['selected_pairs'] != plan['population_pairs'] or
        baseline['observation_pair_count'] != plan['selected_pairs']):
        raise ValueError('frozen sources, checkpoint or fixed population differ')
    weak_rows = {}
    for row in weak['pairs']:
        key = (row['floor_uid'], *sorted((row['observation_a'], row['observation_b'])))
        if key in weak_rows: raise ValueError('duplicate weak-label image pair')
        weak_rows[key] = row
    raw_by_pair, scopes, paths = {}, {}, []
    for uid in sorted({r['floor_uid'] for r in baseline['observation_pairs']}):
        pred_path, audit_path = followup/'predictions'/(uid+'.json'), followup/'audits'/(uid+'.json')
        if content_hash(pred_path) != barrier['prediction_hashes'].get(uid): raise ValueError('prediction barrier mismatch')
        pred, audit = checked(pred_path), checked(audit_path)
        if pred['status'] != 'predicted' or audit['status'] != 'audited' or audit['prediction_fingerprint'] != pred['fingerprint']:
            raise ValueError('prediction or offline audit not valid')
        for row in pred['raw_pairs']:
            key = (uid, row['observation_a'], row['observation_b'])
            if key in raw_by_pair: raise ValueError('duplicate matcher pair')
            raw_by_pair[key] = row
        for row in audit['geometry']['inventory']:
            scopes[(uid, row['pair_id'])] = row['relationship']
        paths.extend((pred_path, audit_path))
    rows = []
    for base in baseline['observation_pairs']:
        key = (base['floor_uid'], base['observation_a'], base['observation_b'])
        if key not in weak_rows or key not in raw_by_pair: raise ValueError('fixed pair missing from source')
        raw = raw_by_pair[key]; scope = scopes.get((base['floor_uid'], raw['pair_id']))
        if scope not in ('same_room', 'cross_room', 'unknown_room'): raise ValueError('pair scope missing')
        rows.append(evaluate_pair(base, weak_rows[key], raw, scope))
    if len(rows) != plan['selected_pairs']: raise ValueError('fixed denominator mismatch')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'pairs.jsonl').open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    report = {'schema': 'research10.r04-fixed-compare.v1', 'status': 'diagnostic_only',
        'fixed_pair_denominator': len(rows), 'cross_room': summarize(rows), 'same_room': summarize(rows, 'same_room'),
        'unknown_room': summarize(rows, 'unknown_room'), 'source_sha256': {str(path.resolve()): content_hash(path) for path in
            [baseline_path, plan_path, weak_path, followup/'run_contract.json', followup/'prediction_barrier.json', followup/'summary.json', *paths]},
        'pairs_sha256': content_hash(output/'pairs.jsonl'),
        'limitations': ['development weak labels, not visually verified; pretraining exposure known',
            'positive denominator includes only pairs with a known positive among predicted candidates; not candidate recall or end-to-end accuracy',
            'forced top-1 raw-score ranking; no dustbin or No-Match threshold and no precision/rejection claim',
            'appearance descriptor contains trained single-view representation, not pure RGB',
            'width comparison assumes shared camera height', 'one seed only; not a trained-module ablation']}
    write_json_atomic(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'fixed_pairs': len(rows), 'cross_room': report['cross_room']}, ensure_ascii=False))


if __name__ == '__main__': main()
