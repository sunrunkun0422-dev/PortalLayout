#!/usr/bin/env python3
"""Export frozen prediction-only PortalLayout hypotheses in the R09 schema.

The frame is PortalLayout's local xz in camera-height units. No claim is made
that it matches SALVe's frame or scale; a separately verified conversion is
required before building a cross-method union.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.research10_candidate_comparison import CANDIDATE_VERSION, PAIR_VERSION, file_hash, matrix
from tools.research10_geometry_followup import checked_artifact


CONTRACT = {'frame_id': 'bi_panorama_local_xz_v1', 'units': 'camera_height_normalized',
            'transform_direction': 'B_to_A', 'equation': 'p_A=T_B_to_A*p_B'}


def export_prediction(data):
    if (data.get('schema_version') != 'research10.geometry-prediction.v1' or
            data.get('ground_truth_read') is not False or data.get('status') != 'predicted'):
        raise ValueError('expected a successful prediction-only artifact')
    floor = data['floor_uid']
    ids = [o['observation_id'] for o in data['observations']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate observation')
    for observation in data['observations']:
        spec = observation.get('prediction', {}).get('coordinate_spec')
        if spec is not None and (spec.get('frame') != 'bi_panorama_local_xz' or
                                 spec.get('units') != 'camera_height_normalized'):
            raise ValueError('unexpected prediction coordinate frame')
    expected = list(itertools.combinations(sorted(ids), 2))
    rows = data['selected_pairs']
    actual = [(row['observation_a'], row['observation_b']) for row in rows]
    if actual != expected or data['pair_denominator'] != len(expected):
        raise ValueError('prediction pair denominator or ordering mismatch')
    pairs, source = [], []
    for row in rows:
        a, b = row['observation_a'], row['observation_b']
        pairs.append({'floor_uid': floor, 'observation_a': a, 'observation_b': b})
        candidates = []
        for h in row['pose_hypotheses']:
            seeds = h['seed_candidate_ids']
            if len(seeds) != 2:
                raise ValueError('invalid seed correspondence')
            try:
                transform = matrix(h['T_B_to_A']).tolist()
                status, reason = 'converted', None
            except (ValueError, TypeError):
                transform, status, reason = None, 'failed', 'invalid_portal_transform'
            candidates.append({'source_candidate_id': h['hypothesis_id'],
                               'source_opening_a': seeds[0], 'source_opening_b': seeds[1],
                               'conversion_status': status, 'reason': reason, 'T_B_to_A': transform})
        source.append({'floor_uid': floor, 'observation_a': a, 'observation_b': b,
                       'candidates': candidates})
    return ({'schema_version': PAIR_VERSION, 'pairs': pairs},
            {'schema_version': CANDIDATE_VERSION, 'method': 'portallayout',
             'coordinate_contract': CONTRACT.copy(), 'pairs': source})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output must be a new directory')
    before = file_hash(args.prediction)
    data = checked_artifact(args.prediction)
    pairs, source = export_prediction(data)
    if before != file_hash(args.prediction): raise ValueError('prediction changed during export')
    args.output.mkdir(parents=True, exist_ok=False)
    details = {'prediction_sha256': before, 'prediction_fingerprint': data['fingerprint'],
               'code_sha256': file_hash(Path(__file__)),
               'pair_denominator': len(pairs['pairs']),
               'candidate_denominator': sum(len(p['candidates']) for p in source['pairs']),
               'scope': 'portal_native_frame_only_common_salve_frame_not_verified'}
    for name, obj in (('pairs.json', pairs), ('portallayout_candidates.json', source), ('provenance.json', details)):
        (args.output/name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    print(json.dumps(details))


if __name__ == '__main__':
    main()
