#!/usr/bin/env python3
"""Freeze an annotation-free pairwise candidate union for R09.

Inputs are already converted to the SAME declared coordinate contract.
This tool does not convert raw SALVe R/t/s, discover data under label-based
directories, run either verifier, or use ground truth.  Candidate and score
files with extra fields are rejected instead of silently carrying labels.
See module constants and tests for the versioned JSON interfaces.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


PAIR_VERSION = 'research10.candidate-pairs.v1'
CANDIDATE_VERSION = 'research10.source-candidates.v1'
SCORE_VERSION = 'research10.candidate-scores.v1'
UNION_VERSION = 'research10.candidate-union.v1'
METHODS = ('portallayout', 'salve_dav2')
ID = re.compile(r'^[A-Za-z0-9_.:-]+$')
FORBIDDEN_ID = ('gt_alignment', 'ground_truth', 'incorrect_alignment', 'exact_gt')


def fields(value, expected, location):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('{} must contain exactly {}'.format(location, sorted(expected)))


def identifier(value, location):
    if (not isinstance(value, str) or not ID.fullmatch(value) or
            any(token in value.lower() for token in FORBIDDEN_ID)):
        raise ValueError('invalid or label-bearing identifier: ' + location)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pair_key(row):
    floor = identifier(row['floor_uid'], 'floor_uid')
    a = identifier(row['observation_a'], 'observation_a')
    b = identifier(row['observation_b'], 'observation_b')
    if a == b: raise ValueError('a candidate pair must contain distinct observations')
    return floor, a, b


def load_pairs(data):
    fields(data, ('schema_version', 'pairs'), 'pair index')
    if data['schema_version'] != PAIR_VERSION or not isinstance(data['pairs'], list):
        raise ValueError('invalid pair index schema')
    pairs, identities = [], set()
    for row in data['pairs']:
        fields(row, ('floor_uid', 'observation_a', 'observation_b'), 'pair')
        key = pair_key(row)
        identity = (key[0], frozenset(key[1:]))
        if identity in identities: raise ValueError('duplicate unordered pair in frozen index')
        pairs.append(key); identities.add(identity)
    if not pairs: raise ValueError('pair index cannot be empty')
    return sorted(pairs)


def matrix(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3, 3) or not np.isfinite(a).all(): raise ValueError('matrix_not_finite_3x3')
    if not np.allclose(a[2], [0., 0., 1.], atol=1e-8, rtol=0.): raise ValueError('invalid_homogeneous_row')
    linear = a[:2, :2]
    determinant = np.linalg.det(linear)
    if not np.isfinite(determinant) or determinant <= 1e-12: raise ValueError('singular_or_reflected_transform')
    gram = linear.T @ linear
    scale2 = np.trace(gram)/2
    if not np.allclose(gram, np.eye(2)*scale2, atol=1e-8, rtol=1e-6):
        raise ValueError('transform_is_not_sim2')
    a[a == 0] = 0.  # Normalize negative zero for stable identity.
    return a


def parse_source(data, expected_method, pairs):
    fields(data, ('schema_version', 'method', 'coordinate_contract', 'pairs'), 'source')
    if data['schema_version'] != CANDIDATE_VERSION or data['method'] != expected_method:
        raise ValueError('invalid source schema or method')
    contract = data['coordinate_contract']
    fields(contract, ('frame_id', 'units', 'transform_direction', 'equation'), 'coordinate contract')
    identifier(contract['frame_id'], 'coordinate frame')
    identifier(contract['units'], 'coordinate units')
    if contract['transform_direction'] != 'B_to_A' or contract['equation'] != 'p_A=T_B_to_A*p_B':
        raise ValueError('candidate direction must be declared as B_to_A')
    if not isinstance(data['pairs'], list): raise ValueError('source pairs must be a list')
    pair_set = set(pairs)
    seen, candidates, failures = set(), [], []
    for row in data['pairs']:
        fields(row, ('floor_uid', 'observation_a', 'observation_b', 'candidates'), 'source pair')
        raw = pair_key(row); reverse = (raw[0], raw[2], raw[1]); key = raw if raw in pair_set else reverse
        if key not in pair_set: raise ValueError('source pair absent from frozen index')
        if key in seen: raise ValueError('duplicate source pair')
        seen.add(key)
        if not isinstance(row['candidates'], list): raise ValueError('candidates must be a list')
        ids = set()
        for item in row['candidates']:
            fields(item, ('source_candidate_id', 'source_opening_a', 'source_opening_b',
                          'conversion_status', 'reason', 'T_B_to_A'), 'source candidate')
            source_id = identifier(item['source_candidate_id'], 'source candidate')
            if source_id in ids: raise ValueError('duplicate source candidate within pair')
            ids.add(source_id)
            opening_a = item['source_opening_a']
            opening_b = item['source_opening_b']
            if opening_a is not None: identifier(opening_a, 'opening_a')
            if opening_b is not None: identifier(opening_b, 'opening_b')
            if raw != key: opening_a, opening_b = opening_b, opening_a
            ref = {'method': expected_method, 'source_candidate_id': source_id,
                   'source_opening_a': opening_a, 'source_opening_b': opening_b}
            status = item['conversion_status']
            if status not in ('converted', 'unsupported', 'failed'):
                raise ValueError('invalid conversion status')
            if status != 'converted':
                if item['T_B_to_A'] is not None: raise ValueError('failed conversion cannot contain a matrix')
                reason = identifier(item['reason'], 'conversion reason')
                failures.append({'pair': list(key), 'source': ref, 'status': status, 'reason': reason})
                continue
            if item['reason'] is not None: raise ValueError('converted candidate cannot have a failure reason')
            try:
                transform = matrix(item['T_B_to_A'])
                if raw != key: transform = matrix(np.linalg.inv(transform))
            except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                detail = str(error) if str(error) in ('matrix_not_finite_3x3', 'invalid_homogeneous_row',
                    'singular_or_reflected_transform', 'transform_is_not_sim2') else type(error).__name__
                failures.append({'pair': list(key), 'source': ref, 'status': 'failed',
                                 'reason': 'invalid_transform:' + detail})
                continue
            candidates.append({'pair': key, 'matrix': transform.tolist(), 'source': ref})
    return contract, seen, candidates, failures


def build_union(pair_index, first, second, *, atol=1e-5, rtol=1e-5):
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError('invalid deduplication tolerance')
    pairs = load_pairs(pair_index)
    first_contract, first_seen, first_rows, first_bad = parse_source(first, METHODS[0], pairs)
    second_contract, second_seen, second_rows, second_bad = parse_source(second, METHODS[1], pairs)
    if first_contract != second_contract: raise ValueError('source coordinate contracts differ')
    all_rows = first_rows + second_rows
    all_rows.sort(key=lambda r: (r['pair'], tuple(np.asarray(r['matrix']).flatten()),
                                 r['source']['method'], r['source']['source_candidate_id']))
    union, groups = [], {}
    for row in all_rows:
        group = groups.setdefault(row['pair'], [])
        existing = next((x for x in group if np.allclose(x['T_B_to_A'], row['matrix'], atol=atol, rtol=rtol)), None)
        if existing is None:
            key = row['pair']
            existing = {'candidate_id': digest([UNION_VERSION, list(key), row['matrix']]),
                        'floor_uid': key[0], 'observation_a': key[1], 'observation_b': key[2],
                        'T_B_to_A': row['matrix'], 'sources': []}
            group.append(existing); union.append(existing)
        existing['sources'].append(row['source'])
    failures = sorted(first_bad + second_bad,
                      key=lambda r: (r['pair'], r['source']['method'], r['source']['source_candidate_id']))
    methods = {}
    for method, seen, converted, bad in ((METHODS[0], first_seen, first_rows, first_bad),
                                         (METHODS[1], second_seen, second_rows, second_bad)):
        methods[method] = {'reported_pairs': len(seen), 'missing_pairs': len(pairs)-len(seen),
                           'missing_pair_ids': [list(p) for p in pairs if p not in seen],
                           'input_candidates': len(converted)+len(bad), 'converted_candidates': len(converted),
                           'failed_or_unsupported_conversions': len(bad)}
    result = {'schema_version': UNION_VERSION, 'coordinate_contract': first_contract,
            'deduplication': {'algorithm': 'sorted_representative_allclose', 'atol': atol, 'rtol': rtol},
            'pair_denominator': len(pairs), 'frozen_pairs': [list(p) for p in pairs],
            'methods': methods, 'union_candidate_denominator': len(union),
            'candidates': union, 'conversion_failures': failures,
            'claim_boundary': 'Declared common frame is not an independently verified coordinate conversion; '
                              'union construction is not a comparison result.'}
    result['union_fingerprint'] = digest(result)
    return result


def score_accounting(union, method, scores=None):
    known = {row['candidate_id'] for row in union['candidates']}
    indexed = {}
    if scores is not None:
        fields(scores, ('schema_version', 'method', 'union_fingerprint', 'scores'), 'scores')
        if (scores['schema_version'] != SCORE_VERSION or scores['method'] != method or
                scores['union_fingerprint'] != union['union_fingerprint']):
            raise ValueError('score schema or method mismatch')
        for row in scores['scores']:
            fields(row, ('candidate_id', 'status', 'score', 'reason'), 'score')
            key = identifier(row['candidate_id'], 'candidate_id')
            if key not in known or key in indexed: raise ValueError('unknown or duplicate scoring candidate')
            if row['status'] == 'scored':
                if (isinstance(row['score'], bool) or not isinstance(row['score'], (int, float)) or
                        not math.isfinite(row['score']) or row['reason'] is not None):
                    raise ValueError('scored candidate requires a finite score and no failure reason')
            elif row['status'] in ('unsupported', 'failed'):
                if row['score'] is not None: raise ValueError('unscored candidate cannot have a score')
                identifier(row['reason'], 'score failure reason')
            else: raise ValueError('invalid score status')
            indexed[key] = row
    rows = [indexed.get(key, {'candidate_id': key, 'status': 'not_reported', 'score': None,
                              'reason': 'no_verifier_output'}) for key in sorted(known)]
    counts = {status: sum(row['status'] == status for row in rows)
              for status in ('scored', 'unsupported', 'failed', 'not_reported')}
    return {'method': method, 'candidate_denominator': len(known), 'counts': counts, 'scores': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', required=True, type=Path)
    p.add_argument('--portal-candidates', required=True, type=Path)
    p.add_argument('--salve-candidates', required=True, type=Path)
    p.add_argument('--portal-scores', type=Path); p.add_argument('--salve-scores', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    sources = [args.pairs, args.portal_candidates, args.salve_candidates]
    sources += [p for p in (args.portal_scores, args.salve_scores) if p is not None]
    if any(any(token in part.lower() for token in FORBIDDEN_ID) for path in sources for part in path.resolve().parts):
        raise ValueError('label-bearing input paths are forbidden')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('output directory must be new or empty')
    if any(output == path.resolve() or output in path.resolve().parents for path in sources):
        raise ValueError('output must not contain any source file')
    before = {str(p.resolve()): file_hash(p) for p in sources}
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    union = build_union(load(args.pairs), load(args.portal_candidates), load(args.salve_candidates))
    union['source_sha256'] = before
    accounting = {method: score_accounting(union, method, load(path) if path else None)
                  for method, path in zip(METHODS, (args.portal_scores, args.salve_scores))}
    if any(file_hash(path) != expected for path, expected in before.items()):
        raise ValueError('source input changed during construction')
    output.mkdir(parents=True, exist_ok=True)
    (output/'candidate_union.json').write_text(json.dumps(union, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (output/'scoring_accounting.json').write_text(json.dumps(accounting, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for method in METHODS:
        with (output/(method+'_scoring_input.jsonl')).open('w', encoding='utf-8') as stream:
            for candidate in union['candidates']:
                stream.write(json.dumps({'union_fingerprint': union['union_fingerprint'], **candidate},
                                        ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'pair_denominator': union['pair_denominator'],
                      'candidate_denominator': union['union_candidate_denominator'],
                      'conversion_failures': len(union['conversion_failures']),
                      'score_counts': {m: accounting[m]['counts'] for m in METHODS}}, indent=2))


if __name__ == '__main__':
    main()
