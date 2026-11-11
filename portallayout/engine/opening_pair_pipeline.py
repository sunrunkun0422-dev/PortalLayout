"""All-unordered-pair inference. No supervision store is read on this path."""
import html
import itertools
from pathlib import Path
import time

import numpy as np
import torch

from portallayout.models.opening_pair_matcher import mutual_matches
from portallayout.utils.frozen_opening_cache import pair_views
from portallayout.utils.opening_representation_adapter import prepare_opening_representation_batch
from portallayout.utils.opening_pair_checkpoint import load_checkpoint
from portallayout.geometry.joint_layout import generate_opening_pose_hypotheses
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def connected_components(observation_ids, edges):
    adjacency = {uid: set() for uid in observation_ids}
    for a, b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    seen, components = set(), []
    for uid in observation_ids:
        if uid in seen: continue
        stack = [uid]; seen.add(uid); group = []
        while stack:
            value = stack.pop(); group.append(value)
            for neighbor in sorted(adjacency[value]-seen):
                seen.add(neighbor); stack.append(neighbor)
        components.append(sorted(group))
    return sorted(components, key=lambda group: (-len(group), group))


def pose_proposals(prediction_a, prediction_b, matches, calibrated=False):
    mapped = []
    for match in matches:
        a, b = prediction_a['openings'][match['index_a']], prediction_b['openings'][match['index_b']]
        if a['geometry_status'] != 'mapped_dominant_wall' or b['geometry_status'] != 'mapped_dominant_wall':
            continue
        mapped.append({'endpoints_a': a['endpoints_local'], 'endpoints_b': b['endpoints_local'],
                       'candidate_id_a': a['candidate_id'], 'candidate_id_b': b['candidate_id']})
    result = []
    for i, seed in enumerate(mapped):
        try:
            hypotheses = generate_opening_pose_hypotheses(seed['endpoints_a'], seed['endpoints_b'],
                independent_matches=mapped[:i]+mapped[i+1:],
                seed_ids=(seed['candidate_id_a'], seed['candidate_id_b']), calibrated=calibrated)
        except ValueError as exc:
            hypotheses = {'status': 'rejected', 'reason': str(exc), 'hypotheses': []}
        hypotheses['seed_candidate_ids'] = [seed['candidate_id_a'], seed['candidate_id_b']]
        result.append(hypotheses)
    return result


def _visualize(path, pair, diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('DIAGNOSTIC: '+pair['training_status']+' | '+pair['status'])
    axes[0, 0].imshow(pair['scores'], aspect='auto'); axes[0, 0].set_title('Opening scores (A x B)')
    axes[0, 1].imshow(pair['conditional_ab'], aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('A to B assignment; final column = unmatched')
    for ax, key, title in ((axes[1, 0], 'attention_ab', 'A opening queries B memory'),
                           (axes[1, 1], 'attention_ba', 'B opening queries A memory')):
        values = diagnostics[key].detach().cpu().numpy()
        ax.imshow(values[0, 0], aspect='auto', extent=[0, 1, 1, 0]); ax.set_title(title+' [0,0]')
        ax.set_xlabel('panorama u'); ax.set_ylabel('panorama v')
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(path, dpi=110); plt.close(fig)


def match_floors(cache, checkpoint, output, *, device='cpu', max_visualizations=12):
    output = Path(output).resolve()
    if output == cache.root or cache.root in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError('use a new inference output outside source cache')
    encoder, matcher, state = load_checkpoint(checkpoint, device)
    if state.get('training_contract', {}).get('source_contract', cache.fingerprint) != cache.fingerprint:
        raise ValueError('checkpoint/frontend contract mismatch')
    encoder.eval(); matcher.eval()
    calibration = state.get('calibration')
    calibrated = bool(state['training_status'] == 'trained' and calibration and
                      calibration.get('weights_hash') == state['weights_hash'] and
                      calibration.get('source') == 'human_reviewed_dev')
    threshold = calibration['threshold'] if calibrated else 0.
    output.mkdir(parents=True, exist_ok=True); (output/'pairs').mkdir(); (output/'review').mkdir()
    start = time.perf_counter()
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    views, encoded_count = {}, 0
    with torch.no_grad():
        for uid in sorted(cache.records):
            record, arrays = cache.load(uid)
            prepared = prepare_opening_representation_batch([record], [arrays], device)
            encoded = encoder(*prepared.encoder_inputs(), return_diagnostics=True)
            views[uid] = pair_views(encoded, prepared)[0]; encoded_count += 1
    results, previews, floor_results = {}, [], []
    for floor in cache.floors:
        refs, edges = [], []
        for aid, bid in itertools.combinations(sorted(floor['observation_order']), 2):
            key = fingerprint([aid, bid])
            if key not in results:
                row = {'pair_id': key, 'observation_a': aid, 'observation_b': bid,
                       'training_status': state['training_status'], 'calibrated': calibrated,
                       'status': 'source_unavailable', 'matches': [], 'pose_proposals': [], 'verified_edge': False}
                if aid in views and bid in views:
                    diagnostic = len(previews) < max_visualizations
                    with torch.no_grad(): result = matcher(views[aid], views[bid], diagnostics=diagnostic)
                    matches = mutual_matches(result, threshold)
                    row.update(status=result['status'], candidate_ids_a=list(views[aid].candidate_ids),
                        candidate_ids_b=list(views[bid].candidate_ids),
                        scores=result['scores'].detach().cpu().tolist(),
                        conditional_ab=result['conditional_ab'].detach().cpu().tolist(),
                        conditional_ba=result['conditional_ba'].detach().cpu().tolist(),
                        iterations=result['iterations'], marginal_error=result['marginal_error'],
                        matches=matches, match_semantics='calibrated' if calibrated else 'diagnostic_only')
                    pa, pb = cache.records[aid]['prediction'], cache.records[bid]['prediction']
                    row['pose_proposals'] = pose_proposals(pa, pb, matches, calibrated)
                    # A rejected/untrained match can still have an explicitly
                    # diagnostic pose proposal, without adding a graph edge.
                    if not matches and result['status'] == 'assigned':
                        s = result['scores']; index = int(s.argmax()); i, j = divmod(index, s.shape[1])
                        proposal = [{'index_a': int(result['indices_a'][i]), 'index_b': int(result['indices_b'][j])}]
                        row['diagnostic_pose_proposals'] = pose_proposals(pa, pb, proposal, False)
                    verified = [h for proposal in row['pose_proposals'] for h in proposal['hypotheses'] if h['status'] == 'verified']
                    if verified:
                        # Several seed openings must support the same transform.
                        matrices = [np.asarray(h['T_B_to_A']) for h in verified]
                        coherent = all(np.allclose(m, matrices[0], rtol=.05, atol=.05) for m in matrices[1:])
                        row['verified_edge'] = calibrated and coherent
                        if coherent: row['T_B_to_A'] = matrices[0].tolist()
                        else: row['pose_conflict'] = True
                    row['unconnected_reason'] = ('verified' if row['verified_edge'] else
                        'model_not_calibrated' if not calibrated else
                        'no_opening_match' if not matches else 'insufficient_or_conflicting_pose_evidence')
                    if diagnostic and 'diagnostics' in result:
                        name = key+'.png'; _visualize(output/'review'/name, row, result['diagnostics'])
                        np.savez_compressed(output/'review'/(key+'.npz'),
                            **{k: v.detach().cpu().numpy() for k, v in result['diagnostics'].items()})
                        previews.append((name, key))
                write_json_atomic(output/'pairs'/(key+'.json'), row)
                results[key] = row
            row = results[key]; refs.append('pairs/'+key+'.json')
            if row['verified_edge']: edges.append((aid, bid))
        components = connected_components(floor['observation_order'], edges)
        record = {'floor_uid': floor['floor_uid'], 'sample_id': floor['sample_id'],
                  'observation_order': floor['observation_order'], 'pairs': refs,
                  'verified_edges': edges, 'components': components,
                  'component_sizes': [len(c) for c in components],
                  'isolated_observations': [c[0] for c in components if len(c) == 1],
                  'global_pose_estimated': False}
        write_json_atomic(output/'floors'/(floor['sample_id']+'.json'), record); floor_results.append(record)
    if device.startswith('cuda'): torch.cuda.synchronize()
    root = Path(__file__).resolve().parents[2]
    report = {'status': 'complete' if all(r['status'] != 'source_unavailable' for r in results.values()) else 'partial',
              'training_status': state['training_status'], 'calibrated': calibrated,
              'source_snapshot': cache.snapshot, 'checkpoint_sha256': content_hash(checkpoint),
              'code_sha256': {p: content_hash(root/p) for p in ('portallayout/models/opening_pair_matcher.py',
                               'portallayout/engine/opening_pair_pipeline.py', 'portallayout/geometry/joint_layout.py')},
              'observations_encoded': encoded_count, 'unordered_pairs': len(results),
              'verified_edges': sum(r['verified_edge'] for r in results.values()),
              'numerical_unconverged': sum(r['status'] == 'numerical_unconverged' for r in results.values()),
              'pair_status_counts': {s: sum(r['status'] == s for r in results.values()) for s in sorted({r['status'] for r in results.values()})},
              'elapsed_seconds': time.perf_counter()-start,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
              'source_files_verified': cache.check_unchanged(), 'ground_truth_read': False,
              'visualizations': len(previews), 'floors': floor_results}
    write_json_atomic(output/'run_report.json', report)
    content = '<meta charset="utf-8"><h1>双图开口匹配诊断</h1><p>'+html.escape(state['training_status'])+'；诊断图不代表匹配已验证。</p>'
    for name, key in previews:
        content += '<p><a href="../pairs/'+key+'.json">候选与位姿证据</a></p><img width="1000" src="'+name+'">'
    (output/'review'/'index.html').write_text(content, encoding='utf-8')
    return report
