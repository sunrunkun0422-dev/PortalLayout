#!/usr/bin/env python3
"""Fixed-candidate appearance and opening-width baselines, without GT or pose.

This is a scoring primitive, not a matcher: no candidate generation, best
choice, rejection threshold, calibration or physical-match conclusion is made.
The caller supplies the exact candidate-ID pairs and all single-view features.

Minimal JSON shape::

  {"schema": "r04-fixed-candidates.v1",
   "view_a": {"view_id":"A", "appearance_source":"predicted_single_view",
      "width_source":"predicted_single_view",
      "scale":{"valid":true,"factor_to_common":1.5,"common_unit":"meters",
               "source":"measured_camera_height"},
      "candidates":[{"candidate_id":"a1","valid":true,
         "appearance":[1,0],"appearance_valid":[true,true],
         "opening_width":0.8,"width_valid":true}]},
   "view_b": { ... },
   "candidate_pairs":[{"candidate_id_a":"a1","candidate_id_b":"b1"}]}

If a width is in camera-height-normalized units, ``factor_to_common`` must
explicitly account for camera height before comparison to meters. An explicitly
stated shared-height assumption can instead use unit ``shared_camera_height``;
it is an assumption, not a measured metric. The source declarations are a
contract for the caller, not proof that data or checkpoint provenance is clean.
"""
import argparse
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


VIEW_KEYS = {'view_id', 'appearance_source', 'width_source', 'scale', 'candidates'}
CANDIDATE_KEYS = {'candidate_id', 'valid', 'appearance', 'appearance_valid', 'opening_width', 'width_valid'}
SCALE_KEYS = {'valid', 'factor_to_common', 'common_unit', 'source'}
PAIR_KEYS = {'candidate_id_a', 'candidate_id_b'}
SCALE_SOURCES = {'measured_camera_height', 'predicted_scale', 'shared_height_assumption', 'unavailable'}
SCALE_UNITS = {'meters', 'shared_camera_height'}


def _boolean(value):
    return isinstance(value, (bool, np.bool_))


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value


def _view(view):
    if not isinstance(view, dict) or set(view) != VIEW_KEYS:
        raise ValueError('view must contain only explicit prediction fields')
    _identifier(view['view_id'], 'view_id')
    if view['appearance_source'] != 'predicted_single_view' or view['width_source'] != 'predicted_single_view':
        raise ValueError('appearance and width must be declared single-view predictions')
    scale = view['scale']
    if not isinstance(scale, dict) or set(scale) != SCALE_KEYS or not _boolean(scale['valid']):
        raise ValueError('scale requires explicit validity, factor, unit and source')
    if not isinstance(scale['source'], str) or scale['source'] not in SCALE_SOURCES:
        raise ValueError('unknown or disallowed scale source')
    if not isinstance(view['candidates'], list):
        raise ValueError('candidates must be a list')
    candidates = {}
    for item in view['candidates']:
        if not isinstance(item, dict) or set(item) != CANDIDATE_KEYS:
            raise ValueError('candidate must contain only prediction fields and explicit masks')
        uid = _identifier(item['candidate_id'], 'candidate_id')
        if uid in candidates:
            raise ValueError('duplicate candidate_id in view')
        if not _boolean(item['valid']) or not _boolean(item['width_valid']):
            raise ValueError('candidate and width validity must be boolean')
        candidates[uid] = item
    return candidates


def validate_input(data):
    if not isinstance(data, dict) or set(data) != {'schema', 'view_a', 'view_b', 'candidate_pairs'}:
        raise ValueError('input must contain only schema, two views and fixed candidate pairs')
    if data['schema'] != 'r04-fixed-candidates.v1':
        raise ValueError('unsupported input schema')
    a, b = _view(data['view_a']), _view(data['view_b'])
    if data['view_a']['view_id'] == data['view_b']['view_id']:
        raise ValueError('view IDs must differ')
    if not isinstance(data['candidate_pairs'], list):
        raise ValueError('candidate_pairs must be a list')
    seen = set()
    for pair in data['candidate_pairs']:
        if not isinstance(pair, dict) or set(pair) != PAIR_KEYS:
            raise ValueError('pair must contain only the two fixed candidate IDs')
        key = (_identifier(pair['candidate_id_a'], 'candidate_id_a'),
               _identifier(pair['candidate_id_b'], 'candidate_id_b'))
        if key in seen:
            raise ValueError('duplicate fixed candidate pair')
        seen.add(key)
    return a, b


def _unknown(reason):
    return {'available': False, 'score': None, 'reason': reason}


def _appearance(item):
    try:
        raw = np.asarray(item['appearance'])
        if raw.dtype.kind not in 'iuf':
            return None, 'appearance_values_unknown'
        x = raw.astype(float, copy=False)
        valid = np.asarray(item['appearance_valid'])
    except (TypeError, ValueError):
        return None, 'invalid_appearance_dimensions'
    if x.ndim != 1 or not len(x) or valid.ndim != 1 or valid.shape != x.shape:
        return None, 'invalid_appearance_dimensions'
    if valid.dtype.kind != 'b':
        return None, 'invalid_appearance_mask'
    if not valid.all() or not np.isfinite(x).all():
        return None, 'appearance_values_unknown'
    maximum = float(np.max(np.abs(x)))
    if maximum <= 0:
        return None, 'zero_appearance_norm'
    # Normalize by the largest absolute value to avoid overflow/underflow.
    scaled = x / maximum
    norm = float(np.linalg.norm(scaled))
    if not math.isfinite(norm) or norm <= 0:
        return None, 'invalid_appearance_norm'
    return scaled / norm, None


def appearance_score(a, b):
    x, error = _appearance(a)
    if error:
        return _unknown('a_' + error)
    y, error = _appearance(b)
    if error:
        return _unknown('b_' + error)
    if x.shape != y.shape:
        return _unknown('appearance_dimensions_mismatch')
    similarity = float(np.clip(np.dot(x, y), -1., 1.))
    return {'available': True, 'score': similarity, 'reason': None}


def _positive_scalar(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _common_width(candidate, scale):
    if not candidate['width_valid']:
        return None, 'width_unknown'
    width = _positive_scalar(candidate['opening_width'])
    if width is None:
        return None, 'invalid_width'
    if not scale['valid']:
        return None, 'scale_unknown'
    factor = _positive_scalar(scale['factor_to_common'])
    unit, source = scale['common_unit'], scale['source']
    if factor is None or not isinstance(unit, str) or unit not in SCALE_UNITS or source == 'unavailable':
        return None, 'invalid_scale'
    if source == 'shared_height_assumption' and unit != 'shared_camera_height':
        return None, 'scale_unit_source_mismatch'
    if source == 'measured_camera_height' and unit != 'meters':
        return None, 'scale_unit_source_mismatch'
    # Use log widths to avoid multiplication overflow in extreme inputs.
    return math.log(width) + math.log(factor), None


def geometry_score(a, b, scale_a, scale_b):
    width_a, error = _common_width(a, scale_a)
    if error:
        return _unknown('a_' + error)
    width_b, error = _common_width(b, scale_b)
    if error:
        return _unknown('b_' + error)
    if scale_a['common_unit'] != scale_b['common_unit']:
        return _unknown('incompatible_common_units')
    if (scale_a['common_unit'] == 'shared_camera_height' and
            (scale_a['source'] != 'shared_height_assumption' or scale_b['source'] != 'shared_height_assumption')):
        return _unknown('incompatible_shared_height_provenance')
    # min(width_a, width_b) / max(width_a, width_b), in [0, 1].
    return {'available': True, 'score': math.exp(-abs(width_a - width_b)), 'reason': None}


def score_fixed_candidates(data):
    """Score every explicitly requested candidate pair without GT or pose."""
    candidates_a, candidates_b = validate_input(data)
    rows = []
    for pair in data['candidate_pairs']:
        ida, idb = pair['candidate_id_a'], pair['candidate_id_b']
        a, b = candidates_a.get(ida), candidates_b.get(idb)
        if a is None or b is None:
            reason = 'missing_candidate_a' if a is None else 'missing_candidate_b'
            appearance, geometry = _unknown(reason), _unknown(reason)
        elif not a['valid'] or not b['valid']:
            reason = 'invalid_candidate_a' if not a['valid'] else 'invalid_candidate_b'
            appearance, geometry = _unknown(reason), _unknown(reason)
        else:
            appearance = appearance_score(a, b)
            geometry = geometry_score(a, b, data['view_a']['scale'], data['view_b']['scale'])
        rows.append({'candidate_id_a': ida, 'candidate_id_b': idb,
                     'appearance_cosine': appearance, 'width_ratio': geometry})
    return {'schema': 'r04-fixed-candidate-scores.v1',
            'view_id_a': data['view_a']['view_id'], 'view_id_b': data['view_b']['view_id'],
            'pair_count': len(rows), 'rows': rows,
            'scoring_only': True, 'calibration': 'not_calibrated',
            'interpretation': 'raw_similarity_not_probability_or_match_decision'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; choose a new file')
    data = json.loads(args.input.read_text(encoding='utf-8'))
    result = score_fixed_candidates(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'pair_count': result['pair_count'],
                      'appearance_available': sum(r['appearance_cosine']['available'] for r in result['rows']),
                      'geometry_available': sum(r['width_ratio']['available'] for r in result['rows'])}))


if __name__ == '__main__':
    main()
