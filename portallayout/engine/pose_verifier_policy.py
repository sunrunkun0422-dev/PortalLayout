"""Prediction-only candidate contract and tentative verifier selection.

Targets, memberships and supervision are deliberately absent from this module.
"""
import copy
import numpy as np
import torch
from portallayout.models.floor_auxiliary import POSE_FEATURE_VERSION
from portallayout.geometry.exploratory_floor_pose import predict_exploratory_pair
from portallayout.geometry.floor_auxiliary_features import auxiliary_features, feature_tensors
from portallayout.utils.room_feature_cache import fingerprint

POLICY_VERSION = 'pose-verifier-candidates.v1'


def canonical_candidates(row):
    """Choose representatives before feature extraction, without model/GT scores."""
    unique = []
    for source in sorted(row['pose_hypotheses'], key=lambda h: (-h.get('match_confidence', 0.),
                           tuple(h['seed_candidate_ids']), h['hypothesis_id'])):
        transform = np.asarray(source['T_B_to_A'], float)
        if not source.get('usable') or not np.isfinite(transform).all(): continue
        scale = np.sqrt(np.linalg.det(transform[:2, :2]))
        if not .5 <= scale <= 2.: continue
        existing = next((h for h in unique if np.allclose(transform, h['T_B_to_A'], atol=1e-5, rtol=1e-5)), None)
        origin = dict(hypothesis_id=source['hypothesis_id'], seed_candidate_ids=source['seed_candidate_ids'])
        if existing is not None:
            existing['sources'].append(origin)
        else:
            h = copy.deepcopy(source); h['sources'] = [origin]; unique.append(h)
    return unique


def feature_for_candidate(a, b, va, vb, row, h):
    match = next(m for m in row['matches'] if [m['candidate_id_a'], m['candidate_id_b']] == h['seed_candidate_ids'])
    # Conditional matrices contain valid candidates only; IDs retain padding.
    ia = list(torch.where(va.valid)[0].cpu().numpy()).index(match['index_a'])
    ib = list(torch.where(vb.valid)[0].cpu().numpy()).index(match['index_b'])
    margin = match['confidence'] - max(row['conditional_ab'][ia][-1], row['conditional_ba'][ib][-1])
    result = auxiliary_features(a, b, va, vb, np.asarray(h['T_B_to_A']), seed_ids=h['seed_candidate_ids'],
                               matches=row['matches'], score=match['confidence'], margin=margin,
                               exclude_duplicates=True)
    # Match the persisted training tensor precision exactly at inference.
    result['visual'] = result['visual'].astype(np.float16).astype(np.float32)
    from portallayout.geometry.pose_geometry import FEATURE_VERSION as GEOMETRY_FEATURE_VERSION
    result['feature_version'] = GEOMETRY_FEATURE_VERSION if row.get('pose_geometry') else POSE_FEATURE_VERSION
    return result


def prepare_pose_pair(a, b, va, vb, bundle, callback=None, raw=None):
    row = copy.deepcopy(raw) if raw is not None else predict_exploratory_pair(a, b, va, vb, bundle)
    row['candidate_policy'] = POLICY_VERSION
    candidates = canonical_candidates(row)
    for h in candidates:
        features = feature_for_candidate(a, b, va, vb, row, h)
        h['nonfit_evidence_available'] = features['independent_evidence']
        h['evidence_supports_pose'] = None  # availability does not certify compatibility
        h['verifier_probability'] = None
        if callback is not None: callback(row, h, features)
        if bundle.auxiliary_trained('pose'):
            if bundle.auxiliary_state['feature_version'] != features['feature_version']:
                raise ValueError('pose filter requires its versioned features')
            with torch.no_grad():
                h['verifier_probability'] = float(bundle.pose_model(*feature_tensors([features], bundle.device)).sigmoid()[0])
    row['pose_hypotheses'] = candidates
    return row


def ranked_hypotheses(row, opening_threshold=.7):
    return sorted([h for h in row['pose_hypotheses'] if h['match_confidence'] >= opening_threshold
                   and h.get('verifier_probability') is not None],
                  key=lambda h: (-h['verifier_probability'], h['hypothesis_id']))


def accepted_best(values, threshold, margin):
    if not values: return None
    best = values[0]
    if best['verifier_probability'] < threshold or not best['nonfit_evidence_available']: return None
    if len(values) > 1 and best['verifier_probability']-values[1]['verifier_probability'] < margin: return None
    return best


def select_verifier_pairs(raw_pairs, policy, workpoint=None, opening_threshold=.7):
    if policy not in ('verifier_rank', 'verifier_filter'): raise ValueError('unknown pose policy')
    if workpoint and workpoint.get('opening_threshold') != opening_threshold: raise ValueError('opening gate differs from workpoint')
    pairs = copy.deepcopy(raw_pairs)
    for row in pairs:
        row.update(pose_status='unverified', T_B_to_A=None, pose_quality=None, calibrated=False,
                   assembly_status='not_used', selection_policy=policy)
        row.pop('selected_hypothesis_id', None)
        values = ranked_hypotheses(row, opening_threshold)
        best = values[0] if policy == 'verifier_rank' and values else None
        if policy == 'verifier_filter' and workpoint and workpoint.get('status') == 'selection_workpoint_found':
            best = accepted_best(values, workpoint['threshold'], workpoint['margin'])
        if best is None:
            row['reason_codes'] = row.get('reason_codes', []) + ['verifier_rejected_or_no_workpoint']
            continue
        row.update(pose_status='tentative', selected_hypothesis_id=best['hypothesis_id'],
                   T_B_to_A=best['T_B_to_A'], pose_quality=max(1e-6, best['verifier_probability']),
                   assembly_status='tentative_eligible', selection_reason=policy,
                   pose_ambiguous=len(values)>1 and best['verifier_probability']-values[1]['verifier_probability'] < .05)
    return pairs


def score_floor(inputs, cache, bundle, config):
    """Persist pure predicted scores once; no offline target store is accepted."""
    import json
    import time
    from pathlib import Path
    from portallayout.engine.floor_layout_pipeline import exploratory_pair_predictions, encode_floor
    from portallayout.utils.room_feature_cache import write_json_atomic
    if not bundle.auxiliary_trained('pose'): raise ValueError('pose verifier requires full trained weights')
    start=time.perf_counter()
    observations,raw,timing=exploratory_pair_predictions(inputs,cache,bundle,config)
    from portallayout.geometry.pose_geometry import geometry_config, overlay_floor, regenerate_pair, FEATURE_VERSION as GEOMETRY_FEATURE_VERSION
    geometry = geometry_config(config) if 'pose_geometry' in config else None
    expected = bundle.auxiliary_state['training_contract'].get('pose_geometry')
    if expected != geometry: raise ValueError('pose geometry/checkpoint contract mismatch')
    feature_version = GEOMETRY_FEATURE_VERSION if geometry else POSE_FEATURE_VERSION
    if geometry:
        observations = overlay_floor(observations, cache, geometry['pose_geometry'])
        lookup = {o['observation_id']:o for o in observations}
        raw = [regenerate_pair(p,lookup[p['observation_a']],lookup[p['observation_b']],geometry) for p in raw]
    binding=dict(policy=POLICY_VERSION,feature_version=feature_version,weights_hash=bundle.weights_hash,
                 source_snapshot=cache.snapshot,raw_fingerprint=fingerprint(raw),floor_uid=inputs['floor_uid'])
    folder=config.get('pose_prediction_cache');path=Path(folder)/(fingerprint(inputs['floor_uid'])+'.json') if folder else None
    if path and path.exists():
        data=json.loads(path.read_text());claimed=data.pop('fingerprint',None)
        if fingerprint(data)!=claimed or data['binding']!=binding:raise ValueError('pose score cache binding mismatch')
        return observations,data['pairs'],dict(encoding_seconds=timing['encoding_seconds'],pair_seconds=time.perf_counter()-start,reused=True)
    _,views,encoding=encode_floor(cache,inputs['observation_order'],bundle)
    lookup={o['observation_id']:o for o in observations}
    pairs=[prepare_pose_pair(lookup[p['observation_a']],lookup[p['observation_b']],views.get(p['observation_a']),
             views.get(p['observation_b']),bundle,raw=p) for p in raw]
    if path:
        payload=dict(binding=binding,pairs=pairs);payload['fingerprint']=fingerprint(payload);write_json_atomic(path,payload)
    return observations,pairs,dict(encoding_seconds=timing['encoding_seconds']+encoding,pair_seconds=time.perf_counter()-start-encoding,reused=False)
