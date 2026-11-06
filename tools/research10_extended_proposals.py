#!/usr/bin/env python3
"""Unverified boundary proposals from one predicted extended layout.

This is an inference-only heuristic: it does not identify which room owns a
wall, detect actual visibility, estimate relative pose, or read annotations.
Use a separate verified matching/visibility stage before local refinement.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from portallayout.geometry.candidate_local_portal_geometry import panorama_ray
from portallayout.utils.room_feature_cache import content_hash, write_json_atomic


def propose(record, arrays, *, minimum_extension=.05, minimum_run_tokens=2):
    """Collect predicted extended depths inside each opening's circular span.

    Depths and minimum_extension are in the source camera-height units. The
    output is a proposal only, regardless of apparent extension or head score.
    Invalid token gaps split proposals; even contiguous points are not treated
    as a verified wall or as evidence that the target room is visible.
    """
    if (not np.isfinite(minimum_extension) or minimum_extension < 0 or
            not isinstance(minimum_run_tokens, int) or minimum_run_tokens < 2):
        raise ValueError('invalid proposal configuration')
    enclosed = np.asarray(arrays['depth_enclosed'], dtype=float).reshape(-1)
    extended = np.asarray(arrays['depth_extended'], dtype=float).reshape(-1)
    if len(enclosed) < 3 or len(enclosed) != len(extended):
        raise ValueError('enclosed and extended token counts must agree')
    uid = record['observation_id']
    prediction = record['prediction']
    spec = prediction['coordinate_spec']
    if spec.get('frame') != 'bi_panorama_local_xz' or spec.get('units') != 'camera_height_normalized':
        raise ValueError('unsupported source coordinate convention')
    output, seen = [], set()
    for opening in prediction.get('openings', []):
        oid = opening.get('candidate_id')
        interval = opening.get('token_interval_inclusive')
        if not isinstance(oid, str) or not oid or oid in seen:
            raise ValueError('opening IDs must be unique nonempty strings')
        seen.add(oid)
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2 or
                any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(enclosed) for v in interval)):
            raise ValueError('invalid opening token interval')
        start, end = interval
        tokens = [(start+i) % len(enclosed) for i in range((end-start) % len(enclosed)+1)]
        runs, current = [], []
        for token in tokens:
            a, b = enclosed[token], extended[token]
            eligible = np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 and b-a >= minimum_extension
            if eligible:
                current.append(token)
            else:
                if len(current) >= minimum_run_tokens: runs.append(current)
                current = []
        if len(current) >= minimum_run_tokens: runs.append(current)
        candidates = []
        for indices in runs:
            points = [list(map(float, panorama_ray(index, len(enclosed))*extended[index])) for index in indices]
            candidates.append({'token_indices': indices, 'boundary_points': points,
                               'extension_depth': [float(extended[i]-enclosed[i]) for i in indices]})
        output.append({'source_observation_id': uid, 'source_opening_id': oid,
                       'source_opening_tokens': list(interval), 'token_count': len(tokens),
                       'runs': candidates, 'status': 'unverified_candidate' if candidates else 'no_candidate',
                       'visibility': 'unknown', 'target_membership': 'unknown',
                       'source': 'predicted_extended_depth_inside_opening',
                       'coordinate_spec': dict(spec)})
    return {'schema_version': 'research10.extended-proposals.v1', 'observation_id': uid,
            'minimum_extension_height_units': minimum_extension,
            'minimum_run_tokens': minimum_run_tokens, 'openings': output,
            'uses_ground_truth': False, 'accepted_for_refinement': False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-cache', required=True)
    p.add_argument('--observation-id', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    path = Path(args.output)
    if path.exists(): raise ValueError('output must be new')
    from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
    cache = FrozenOpeningCache(args.prediction_cache)
    record, arrays = cache.load(args.observation_id)
    result = propose(record, arrays)
    result['source_snapshot'] = cache.snapshot
    result['source_contract'] = cache.fingerprint
    result['source_cache_key'] = record['cache_key']
    result['source_record_sha256'] = content_hash(cache.root / next(
        r['metadata_file'] for floor in cache.floors for r in floor['observations']
        if r['observation_id'] == args.observation_id and r['status'] == 'predicted'))
    cache.check_unchanged()
    write_json_atomic(path, result)
    print(json.dumps({'observation_id': args.observation_id, 'openings': len(result['openings']),
                      'proposed_openings': sum(bool(x['runs']) for x in result['openings']),
                      'accepted_for_refinement': False}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
