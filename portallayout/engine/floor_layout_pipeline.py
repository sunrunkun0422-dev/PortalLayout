"""Prediction-only whole-floor orchestration. GT access lives in separate tools."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from portallayout.models.floor_auxiliary import PoseHypothesisVerifier, SameSpaceAssociator, FEATURE_VERSION, SUPPORTED_FEATURE_VERSIONS
from portallayout.models.opening_pair_matcher import mutual_matches
from portallayout.utils.opening_pair_checkpoint import load_checkpoint
from portallayout.engine.opening_pair_pipeline import pose_proposals
from portallayout.utils.frozen_opening_cache import FrozenOpeningCache, pair_views
from portallayout.utils.opening_representation_adapter import prepare_opening_representation_batch
from portallayout.geometry.floor_auxiliary_features import auxiliary_features, feature_tensors
from portallayout.geometry.floor_geometry import local_scale
from portallayout.geometry.floor_pose_graph import optimize_floor_graph, path_transform, edge_residual, PoseGraphConfig
from portallayout.geometry.floor_assembly import assemble_floor, pair_edges
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def bundle_weights_hash(pair_path, auxiliary_path=None):
    return fingerprint({'pair': content_hash(pair_path), 'auxiliary': content_hash(auxiliary_path) if auxiliary_path else None,
                        'feature_version': FEATURE_VERSION})


class FloorModelBundle:
    def __init__(self, specification, device='cpu'):
        if isinstance(specification, (str, Path)):
            path = Path(specification).resolve(); specification = json.loads(path.read_text())
            specification = dict(specification)
            for key in ('pair_checkpoint', 'auxiliary_checkpoint', 'calibration'):
                if specification.get(key): specification[key] = str((path.parent/specification[key]).resolve())
        self.spec = dict(specification); self.device = device
        self.encoder, self.matcher, self.pair_state = load_checkpoint(self.spec['pair_checkpoint'], device)
        self.encoder.eval(); self.matcher.eval()
        self.pose_model, self.room_model = PoseHypothesisVerifier().to(device), SameSpaceAssociator().to(device)
        self.auxiliary_status = 'initialized_untrained'
        self.auxiliary_state = None
        if self.spec.get('auxiliary_checkpoint'):
            state = torch.load(self.spec['auxiliary_checkpoint'], map_location='cpu')
            if state.get('feature_version') not in SUPPORTED_FEATURE_VERSIONS: raise ValueError('auxiliary feature contract mismatch')
            if state.get('pair_checkpoint_sha256') != content_hash(self.spec['pair_checkpoint']):
                raise ValueError('auxiliary checkpoint belongs to different pair weights')
            self.pose_model.load_state_dict(state['pose_state'], strict=True)
            self.room_model.load_state_dict(state['room_state'], strict=True)
            self.auxiliary_status = state['training_status']; self.auxiliary_state = state
        self.pose_model.eval(); self.room_model.eval()
        self.weights_hash = bundle_weights_hash(self.spec['pair_checkpoint'], self.spec.get('auxiliary_checkpoint'))
        self.calibration = json.loads(Path(self.spec['calibration']).read_text()) if self.spec.get('calibration') else {}
        if self.calibration:
            if self.calibration.get('weights_hash') != self.weights_hash or self.calibration.get('feature_version') != FEATURE_VERSION:
                raise ValueError('calibration/weights/feature fingerprint mismatch')
            if self.calibration.get('track') not in ('geometry_dev', 'human_reviewed_dev'):
                raise ValueError('unknown calibration provenance')

    def auxiliary_trained(self, stage):
        state = self.auxiliary_state
        return bool(state and state.get('training_contract', {}).get('mode') == 'full' and
                    state.get('reports', {}).get(stage, {}).get('status') in ('passed', 'diagnostic_loss_not_decreased'))

    def calibrated(self, stage):
        return bool(self.pair_state['training_status'] == 'trained' and
                    (stage == 'opening' or self.auxiliary_trained(stage)) and
                    self.calibration.get(stage, {}).get('status') == 'calibrated')


def encode_floor(cache, ids, bundle):
    views, observations = {}, []
    start = time.perf_counter()
    with torch.no_grad():
        for uid in sorted(ids):
            if uid not in cache.records:
                observations.append({'observation_id': uid, 'input_status': 'prediction_unavailable', 'prediction': {}}); continue
            record, arrays = cache.load(uid)
            prepared = prepare_opening_representation_batch([record], [arrays], bundle.device)
            encoded = bundle.encoder(*prepared.encoder_inputs(), return_diagnostics=True)
            views[uid] = pair_views(encoded, prepared)[0]
            observations.append({'observation_id': uid, 'input_status': 'predicted', 'source_cache_key': record['cache_key'],
                                 'prediction': record['prediction']})
    return observations, views, time.perf_counter()-start


def predict_pair(a, b, va, vb, bundle, *, auxiliary=True, feature_callback=None):
    aid, bid = a['observation_id'], b['observation_id']; pair_id = fingerprint([aid, bid])
    row = {'pair_id': pair_id, 'observation_a': aid, 'observation_b': bid, 'match_status': 'insufficient_input',
           'pose_status': 'unverified', 'assembly_status': 'not_used', 'matches': [], 'pose_hypotheses': [],
           'T_B_to_A': None, 'pose_quality': None, 'calibrated': False, 'reason_codes': [],
           'confidence_source': bundle.calibration.get('track', 'diagnostic_only')}
    if va is None or vb is None:
        row['reason_codes'].append('prediction_unavailable'); return row
    with torch.no_grad(): result = bundle.matcher(va, vb)
    row.update(numerical_status=result['status'], candidate_ids_a=list(va.candidate_ids), candidate_ids_b=list(vb.candidate_ids))
    for key in ('scores', 'conditional_ab', 'conditional_ba'):
        value = result.get(key)
        row[key] = value.detach().cpu().tolist() if torch.is_tensor(value) else value
    row['sinkhorn'] = {key: result[key] for key in ('iterations', 'marginal_error', 'converged')}
    row['assignment_method'] = result.get('assignment_method', 'log_sinkhorn')
    if row['assignment_method'] != 'log_sinkhorn':
        row['sinkhorn'] = {'applicable': False, 'reason': 'legacy uses bidirectional softmax'}
    if result['status'] != 'assigned':
        row['match_status'] = result['status']; row['reason_codes'].append(result['status']); return row
    calibrated = bundle.calibrated('opening')
    threshold = bundle.calibration['opening']['threshold'] if calibrated else 0.
    # Auxiliary feature evidence uses a fixed retrieval policy in offline
    # export and inference. The calibrated opening gate selects seed hypotheses,
    # but must not silently change the feature distribution being calibrated.
    proposal_matches = mutual_matches(result, 0.)
    for match in proposal_matches:
        match['candidate_id_a'] = va.candidate_ids[match['index_a']]
        match['candidate_id_b'] = vb.candidate_ids[match['index_b']]
        ia = int(torch.where(result['indices_a'] == match['index_a'])[0][0])
        ib = int(torch.where(result['indices_b'] == match['index_b'])[0][0])
        match['dustbin_margin'] = match['confidence']-max(float(result['conditional_ab'][ia, -1]),
                                                        float(result['conditional_ba'][ib, -1]))
    matches = [m for m in proposal_matches if m['confidence'] >= threshold]
    row.update(matches=matches, calibrated=calibrated,
               match_status=('accepted' if matches else 'no_match') if calibrated else 'diagnostic_only')
    if not calibrated: row['reason_codes'].append('opening_model_not_calibrated')
    proposals = pose_proposals(a['prediction'], b['prediction'], matches, calibrated=calibrated)
    candidates = []
    for proposal in proposals:
        seed_ids = proposal['seed_candidate_ids']
        match = next(x for x in proposal_matches if [x['candidate_id_a'], x['candidate_id_b']] == seed_ids)
        if calibrated and match['confidence'] < threshold: continue
        score = float(match.get('confidence', match.get('score', 0.)))
        for hypothesis in proposal['hypotheses']:
            transform = np.array(hypothesis['T_B_to_A'])
            h = dict(hypothesis, hypothesis_id=fingerprint([pair_id, seed_ids, hypothesis['endpoint_mapping']]),
                     seed_candidate_ids=seed_ids, verifier_probability=None, independent_feature_evidence=False)
            features = auxiliary_features(a, b, va, vb, transform, seed_ids=seed_ids, matches=proposal_matches,
                                          score=score, margin=match['dustbin_margin'])
            h['independent_feature_evidence'] = features['independent_evidence']
            if bundle.auxiliary_status != 'initialized_untrained':
                with torch.no_grad(): h['verifier_probability'] = float(bundle.pose_model(*feature_tensors([features], bundle.device)).sigmoid()[0])
            if feature_callback is not None:
                feature_callback('pose', row, h, features)
                room_features = auxiliary_features(a, b, va, vb, transform, task='room')
                feature_callback('room', row, h, room_features)
            candidates.append(h)
    # Equivalent transforms from several seed openings are one hypothesis, not competitors.
    deduplicated = []
    for h in sorted(candidates, key=lambda h: (-(h['verifier_probability'] or 0.), h['hypothesis_id'])):
        equivalent = next((v for v in deduplicated if np.allclose(v['T_B_to_A'], h['T_B_to_A'], atol=1e-5, rtol=1e-5)), None)
        if equivalent is None:
            h['supporting_seed_ids'] = [h['seed_candidate_ids']]; deduplicated.append(h)
        else:
            equivalent['supporting_seed_ids'].append(h['seed_candidate_ids'])
            equivalent['supported'] = equivalent.get('supported', False) or h.get('supported', False)
    row['pose_hypotheses'] = deduplicated
    if not candidates:
        row['reason_codes'].append('no_usable_opening_geometry' if matches else 'no_accepted_correspondence'); return row
    selected = None
    if calibrated and auxiliary and bundle.calibrated('pose'):
        settings = bundle.calibration['pose']
        eligible = [h for h in deduplicated if h['verifier_probability'] is not None]
        if eligible:
            best = eligible[0]; second = eligible[1]['verifier_probability'] if len(eligible) > 1 else 0.
            if best['verifier_probability'] >= settings['threshold'] and best['independent_feature_evidence']:
                if best['verifier_probability']-second >= settings.get('margin', 0.): selected = best
                else: row['pose_status'] = 'ambiguous'
    elif calibrated and not auxiliary:
        supported = [h for h in deduplicated if h.get('supported')]
        if len(supported) == 1:
            selected = supported[0]; row['confidence_source'] = 'geometry_dev_held_out_openings'
        elif len(supported) > 1: row['pose_status'] = 'ambiguous'
    else: row['reason_codes'].append('pose_model_not_calibrated')
    if selected is not None:
        accept_hypothesis(row, selected, float(selected.get('verifier_probability') or min(x.get('confidence', 1.) for x in matches)))
    elif row['pose_status'] != 'ambiguous': row['reason_codes'].append('insufficient_independent_pose_evidence')
    return row


def accept_hypothesis(row, hypothesis, quality):
    row.update(pose_status='accepted', T_B_to_A=hypothesis['T_B_to_A'], pose_quality=max(min(quality, 1.), 1e-6),
               selected_hypothesis_id=hypothesis['hypothesis_id'], assembly_status='eligible')


def _edges(pairs):
    return [dict(edge_id=p['pair_id'], a=p['observation_a'], b=p['observation_b'],
                 T_B_to_A=p['T_B_to_A'], weight=p['pose_quality'], confidence_source=p['confidence_source'])
            for p in pairs if p['pose_status'] == 'accepted' and p['calibrated']]


def exploratory_pair_predictions(inputs, cache, bundle, config):
    """Cache only prediction artifacts; fail closed on model/input/schema changes."""
    from portallayout.geometry.exploratory_floor_pose import VERSION, predict_exploratory_pair
    ids = sorted(inputs['observation_order'])
    binding = dict(schema_version='floor-pair-predictions.v1', algorithm=VERSION,
                   weights_hash=bundle_weights_hash(bundle.spec['pair_checkpoint']), frontend_contract=cache.fingerprint,
                   floor_uid=inputs['floor_uid'], observation_order=ids, observations={})
    for uid in ids:
        r = cache.records.get(uid)
        if r is not None:
            binding['observations'][uid] = dict(cache_key=r['cache_key'], prediction=fingerprint(r['prediction']),
                candidate_ids=[o['candidate_id'] for o in r['prediction']['openings']],
                tensor_sha256=cache.before[str(cache.root/r['tensor_file'])])
    folder = config.get('pair_prediction_cache')
    path = Path(folder)/(fingerprint(inputs['floor_uid'])+'.json') if folder else None
    if path and path.exists():
        payload = json.loads(path.read_text()); claimed = payload.pop('fingerprint', None)
        if fingerprint(payload) != claimed or payload.get('binding') != binding:
            raise ValueError('pure prediction pair cache fingerprint or binding mismatch')
        pairs = payload['pairs']
        expected = [(a, b) for a, b in itertools.combinations(ids, 2)]
        if [(p['observation_a'], p['observation_b']) for p in pairs] != expected:
            raise ValueError('pure prediction pair cache coverage mismatch')
        observations = [dict(observation_id=uid, input_status='predicted' if uid in cache.records else 'prediction_unavailable',
            prediction=cache.records[uid]['prediction'] if uid in cache.records else {},
            source_cache_key=cache.records[uid]['cache_key'] if uid in cache.records else None) for uid in ids]
        return observations, pairs, dict(encoding_seconds=0., pair_seconds=0., reused=True)
    observations, views, encoding_seconds = encode_floor(cache, ids, bundle)
    lookup = {r['observation_id']: r for r in observations}; pairs = []; start = time.perf_counter()
    for a, b in itertools.combinations(ids, 2):
        pairs.append(predict_exploratory_pair(lookup[a], lookup[b], views.get(a), views.get(b), bundle))
    seconds = time.perf_counter()-start
    if path:
        payload = dict(binding=binding, pairs=pairs)
        payload['fingerprint'] = fingerprint(payload); write_json_atomic(path, payload)
    return observations, pairs, dict(encoding_seconds=encoding_seconds, pair_seconds=seconds, reused=False)


def resolve_with_independent_paths(pairs, observations, bundle, graph_config):
    if not bundle.calibrated('pose'): return
    scales = {r['observation_id']: local_scale(r) for r in observations}
    settings = bundle.calibration['pose']
    for iteration in range(3):
        edges = _edges(pairs); added = False
        for row in sorted(pairs, key=lambda r: r['pair_id']):
            if row['pose_status'] == 'accepted' or not row['calibrated']: continue
            aid, bid = row['observation_a'], row['observation_b']
            path, ids = path_transform(edges, aid, bid, [row['pair_id']])
            if path is None: continue
            compatible = []
            for h in row['pose_hypotheses']:
                if h.get('verifier_probability') is None or h['verifier_probability'] < settings['threshold']: continue
                e = dict(a=aid, b=bid, T_B_to_A=h['T_B_to_A'])
                if np.linalg.norm(edge_residual(e, {aid: np.eye(3), bid: path}, scales, graph_config)) <= graph_config.conflict_sigma:
                    compatible.append(h)
            if len(compatible) == 1:
                accept_hypothesis(row, compatible[0], compatible[0]['verifier_probability'])
                row['independent_path_edge_ids'] = ids; row['reason_codes'].append('resolved_by_independent_path'); added = True
        if not added: break


def estimate_floor(inputs, model_bundle, config=None, cache=None):
    config = dict(config or {}); start = time.perf_counter()
    mode = config.get('inference_mode', 'strict')
    if mode not in ('strict', 'exploratory'): raise ValueError('unknown inference mode')
    if mode == 'strict' and config.get('room_association') == 'geometry':
        raise ValueError('geometry room association requires exploratory mode')
    frontend_seconds = 0.
    bundle = model_bundle if isinstance(model_bundle, FloorModelBundle) else FloorModelBundle(model_bundle, config.get('device', 'cpu'))
    if bundle.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    if cache is None:
        if not config.get('frontend_cache_output') or not config.get('raw_root'):
            raise ValueError('RGB inference requires raw_root and a new frontend_cache_output')
        from portallayout.engine.floor_observation_pipeline import (ObservationPredictionSettings, frontend_contract,
            BiLayoutObservationEncoder, predict_floor_observations)
        from portallayout.utils.room_feature_cache import ObservationFeatureCache
        frontend = json.loads(Path(config.get('frontend_config', 'configs/research1_floor_frontend.json')).read_text())
        settings = ObservationPredictionSettings(**frontend['prediction'])
        paths = [frontend[k] for k in ('bi_layout_config', 'bi_layout_checkpoint', 'opening_checkpoint')]
        contract = frontend_contract(*paths, settings, execution_device=bundle.device)
        writable = ObservationFeatureCache(config['frontend_cache_output'], contract, resume=config.get('resume_cache', False))
        encoder = BiLayoutObservationEncoder.from_checkpoints(*paths, settings, bundle.device)
        predict_floor_observations([inputs], config['raw_root'], encoder, writable, settings)
        del encoder
        cache = FrozenOpeningCache(writable.root)
        frontend_seconds = time.perf_counter()-start
    elif not isinstance(cache, FrozenOpeningCache): cache = FrozenOpeningCache(cache)
    if bundle.pair_state.get('training_contract', {}).get('source_contract', cache.fingerprint) != cache.fingerprint:
        raise ValueError('frontend/pair model contract mismatch')
    views = {}; reused = False
    pose_policy = config.get('pose_policy', 'geometry')
    if 'pose_geometry' in config and (mode != 'exploratory' or pose_policy == 'geometry'):
        raise ValueError('geometry overlay requires an exploratory verifier policy')
    if pose_policy not in ('geometry', 'verifier_rank', 'verifier_filter'): raise ValueError('unknown pose policy')
    if pose_policy != 'geometry' and mode != 'exploratory': raise ValueError('learned experimental policy requires exploratory mode')
    if mode == 'exploratory' and pose_policy != 'geometry':
        from portallayout.engine.pose_verifier_policy import score_floor
        observations, pairs, timing = score_floor(inputs, cache, bundle, config)
        encoding_seconds, pair_seconds, reused = timing['encoding_seconds'], timing['pair_seconds'], timing['reused']
    elif mode == 'exploratory' or (config.get('pair_prediction_cache') and not bundle.calibrated('opening') and
                                not bundle.auxiliary_trained('pose')):
        observations, pairs, timing = exploratory_pair_predictions(inputs, cache, bundle, config)
        encoding_seconds, pair_seconds, reused = timing['encoding_seconds'], timing['pair_seconds'], timing['reused']
    else:
        observations, views, encoding_seconds = encode_floor(cache, inputs['observation_order'], bundle)
        lookup = {r['observation_id']: r for r in observations}
        pairs, pair_start = [], time.perf_counter()
        for aid, bid in itertools.combinations(sorted(inputs['observation_order']), 2):
            pairs.append(predict_pair(lookup[aid], lookup[bid], views.get(aid), views.get(bid), bundle,
                                     auxiliary=config.get('auxiliary_verification', True)))
        pair_seconds = time.perf_counter()-pair_start
    lookup = {r['observation_id']: r for r in observations}
    pair_sink = config.get('pair_output')
    if pair_sink: Path(pair_sink).mkdir(parents=True, exist_ok=True)
    graph_start = time.perf_counter()
    graph_config = PoseGraphConfig(**config.get('pose_graph', {'mode': config.get('pose_opt', 'robust'),
                                                             'sparse_jacobian': mode == 'exploratory'}))
    if mode == 'exploratory' and pose_policy != 'geometry':
        from portallayout.engine.pose_verifier_policy import select_verifier_pairs, POLICY_VERSION
        workpoint = config.get('pose_workpoint')
        if isinstance(workpoint, (str, Path)): workpoint = json.loads(Path(workpoint).read_text())
        if workpoint is not None:
            checked = dict(workpoint); claimed = checked.pop('fingerprint', None)
            if fingerprint(checked) != claimed or checked.get('weights_hash') != bundle.weights_hash or checked.get('candidate_policy') != POLICY_VERSION or checked.get('source_snapshot') != cache.snapshot:
                raise ValueError('pose workpoint binding mismatch')
        pairs = select_verifier_pairs(pairs, pose_policy, workpoint, config.get('opening_threshold', .7))
    elif mode == 'exploratory':
        from portallayout.geometry.exploratory_floor_pose import select_exploratory_poses
        pairs = select_exploratory_poses(pairs, observations, config.get('opening_threshold', .5), graph_config)
    elif config.get('auxiliary_verification', True): resolve_with_independent_paths(pairs, observations, bundle, graph_config)
    graph = optimize_floor_graph(inputs['observation_order'], pair_edges(pairs, mode), {uid: local_scale(r) for uid, r in lookup.items()}, graph_config)
    rejected = set(graph['rejected_edges'])
    for p in pairs:
        if p['pair_id'] in rejected: p['assembly_status'] = 'cycle_conflict'
        elif p['pose_status'] in ('accepted', 'tentative'): p['assembly_status'] = 'tentative_used' if mode == 'exploratory' else 'used'
    graph_seconds = time.perf_counter()-graph_start
    room_start = time.perf_counter(); relationships = []
    if mode == 'exploratory' and config.get('room_fusion', True) and config.get('room_association') == 'geometry':
        from portallayout.geometry.exploratory_floor_pose import geometric_room_relationships
        relationships = geometric_room_relationships(observations, graph)
    elif mode == 'strict' and config.get('room_fusion', True) and bundle.calibrated('room'):
        with torch.no_grad():
            for c in graph['components']:
                for aid, bid in itertools.combinations(c['observation_ids'], 2):
                    if aid not in views or bid not in views: continue
                    transform = np.linalg.inv(np.array(c['poses'][aid])) @ np.array(c['poses'][bid])
                    f = auxiliary_features(lookup[aid], lookup[bid], views[aid], views[bid], transform, task='room')
                    probability = float(bundle.room_model(*feature_tensors([f], bundle.device)).sigmoid()[0])
                    relationships.append({'a': aid, 'b': bid, 'probability': probability,
                                          'valid': bool(f['geometry'][:, -1].all()), 'source': 'predicted_geometry_and_visual'})
    result = assemble_floor(observations, pairs, dict(config, floor_uid=inputs['floor_uid'], _precomputed_graph=graph,
        room_relationships=relationships, room_threshold=bundle.calibration.get('room', {}).get('threshold', 1.)))
    result['provenance'] = {'weights_hash': bundle.weights_hash, 'source_snapshot': cache.snapshot,
        'inference_mode': mode, 'experimental_opening_threshold': config.get('opening_threshold', .5) if mode == 'exploratory' else None,
        'room_association': config.get('room_association', 'learned'), 'pure_prediction_cache_reused': reused,
        'pose_policy': pose_policy, 'pose_workpoint_fingerprint': workpoint.get('fingerprint') if pose_policy != 'geometry' and workpoint else None,
        'pose_geometry': bundle.auxiliary_state.get('training_contract', {}).get('pose_geometry') if bundle.auxiliary_state else None,
        'matcher_type': bundle.pair_state.get('matcher_type', 'new'),
        'frontend_contract': cache.fingerprint, 'pair_training_status': bundle.pair_state['training_status'],
        'auxiliary_training_status': bundle.auxiliary_status, 'calibration_track': bundle.calibration.get('track'),
        'calibration_status': {s: bundle.calibrated(s) for s in ('opening', 'pose', 'room')}}
    result['runtime'] = {'frontend_seconds': frontend_seconds, 'encoding_seconds': encoding_seconds, 'pair_seconds': pair_seconds,
        'graph_seconds': graph_seconds, 'room_and_assembly_seconds': time.perf_counter()-room_start,
        'total_seconds': time.perf_counter()-start,
        'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if bundle.device.startswith('cuda') else None,
        'observations_encoded': 0 if reused else sum(r['input_status'] == 'predicted' for r in observations), 'unordered_pairs': len(pairs)}
    if frontend_seconds: result['provenance']['source_files_verified'] = cache.check_unchanged()
    if pair_sink:
        for row in pairs: write_json_atomic(Path(pair_sink)/(row['pair_id']+'.json'), row)
    return result


def failed_floor(inputs, error):
    """Retain failed inputs in the evaluation denominator without inventing poses."""
    ids = sorted(inputs['observation_order'])
    reason = '{}: {}'.format(type(error).__name__, error)
    pairs = [dict(pair_id=fingerprint([a, b]), observation_a=a, observation_b=b,
                  match_status='runtime_failure', pose_status='unverified', calibrated=False,
                  matches=[], pose_hypotheses=[], reason_codes=['floor_runtime_failure'], error=reason)
             for a, b in itertools.combinations(ids, 2)]
    return dict(floor_uid=inputs['floor_uid'], status='runtime_failure', error=reason,
                observation_order=ids, observations=[dict(observation_id=uid, pose_status='unavailable') for uid in ids],
                components=[], main_component_id=None, rooms=[], openings=[], structural_graphs={},
                pair_results=pairs, adjacency={'observation_order': ids, 'status': 'unavailable'},
                coordinate_spec={'units': 'camera_height_normalized', 'scale_status': 'unknown'},
                provenance={'ground_truth_read': False}, runtime={'unordered_pairs': len(pairs)})


def export_floor(prediction, output):
    from portallayout.visualization.floor_layout import render_floor
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    render_floor(prediction, output)
    prediction['runtime']['export_render_seconds'] = time.perf_counter()-start
    compact = {k: v for k, v in prediction.items() if k not in ('pair_results',)}
    compact['pair_results_file'] = 'pair_results.json'
    write_json_atomic(output/'floor_result.json', compact)
    write_json_atomic(output/'pair_results.json', {'pairs': prediction['pair_results']})
    write_json_atomic(output/'adjacency_pred.json', prediction['adjacency'])
    write_json_atomic(output/'components.json', {'main_component_id': prediction['main_component_id'], 'components': prediction['components']})
    reasons = {}
    for pair in prediction['pair_results']:
        for code in pair['reason_codes']: reasons[code] = reasons.get(code, 0)+1
    report = '# Floor reconstruction\n\nStatus: '+prediction['status']+'\n\n'
    report += 'Coordinates: '+json.dumps(prediction['coordinate_spec'])+'\n\n'
    report += 'Inputs: %d; pairs: %d; components: %s; predicted rooms: %d.\n\n'%(
        len(prediction['observations']), len(prediction['pair_results']),
        [len(c['observation_ids']) for c in prediction['components']], len(prediction['rooms']))
    report += 'Independent components have unknown relative positions. Sidebar placement is display-only.\n\n'
    report += 'Calibration and training:\n```json\n'+json.dumps(prediction['provenance'], indent=2)+'\n```\n\n'
    report += 'Reasons:\n```json\n'+json.dumps(reasons, indent=2)+'\n```\n\n'
    report += 'Runtime:\n```json\n'+json.dumps(prediction['runtime'], indent=2)+'\n```\n'
    (output/'floor_report.md').write_text(report)
    return str(output/'floor_result.json')
