"""Assemble fixed local predictions; room association never consumes GT IDs."""
import copy
import itertools
import numpy as np
from shapely.ops import unary_union
from portallayout.geometry.floor_geometry import observation_polygon, transform_shape, geojson, local_scale, partition_graph, points_transform
from portallayout.geometry.floor_pose_graph import PoseGraphConfig, optimize_floor_graph
from portallayout.utils.room_feature_cache import fingerprint


def pair_edges(pairs, inference_mode='strict'):
    if inference_mode not in ('strict', 'exploratory'): raise ValueError('invalid inference mode')
    edges = []
    for pair in pairs:
        status = pair.get('pose_status')
        if status == 'tentative':
            if inference_mode != 'exploratory' or pair.get('calibrated', False):
                raise ValueError('tentative edge requires exploratory mode and must remain uncalibrated')
        elif status == 'accepted':
            if not pair.get('calibrated', False): raise ValueError('uncalibrated edge cannot enter assembly')
        else: continue
        edges.append(dict(edge_id=pair['pair_id'], a=pair['observation_a'], b=pair['observation_b'],
                          T_B_to_A=pair['T_B_to_A'], weight=pair['pose_quality'],
                          confidence_source=pair['confidence_source'], validation_status=status))
    return edges


def complete_link_rooms(ids, relationships, threshold):
    groups = [[uid] for uid in sorted(ids)]
    scores = {}
    for row in relationships:
        key = tuple(sorted((row['a'], row['b'])))
        value = float(row['rule_passed']) if 'rule_passed' in row else row.get('probability')
        valid = row.get('valid', False) and value is not None and np.isfinite(value) and 0 <= value <= 1
        # Unknown/conflicting duplicate evidence must not be overwritten by a
        # later positive row. Use the conservative value in either input order.
        value = float(value) if valid else None
        if key in scores:
            value = min(scores[key], value) if scores[key] is not None and value is not None else None
        scores[key] = value
    while True:
        proposals = []
        for i, a in enumerate(groups):
            for j in range(i+1, len(groups)):
                b = groups[j]; values = [scores.get(tuple(sorted((x, y)))) for x in a for y in b]
                if all(v is not None and v >= threshold for v in values):
                    proposals.append((-min(values), tuple(sorted(a+b)), i, j))
        if not proposals: break
        _, merged, i, j = min(proposals)
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]+[list(merged)]
        groups.sort()
    return groups


def assemble_floor(observations, pair_predictions, config=None):
    config = dict(config or {})
    ids = [r['observation_id'] for r in observations]
    if len(ids) != len(set(ids)): raise ValueError('duplicate observation identity')
    rows = {r['observation_id']: copy.deepcopy(r) for r in observations}
    mode = config.get('inference_mode', 'strict')
    exploratory = mode == 'exploratory'
    edges = pair_edges(pair_predictions, mode)
    graph_config = PoseGraphConfig(**config.get('pose_graph', {'mode': config.get('pose_opt', 'robust')}))
    graph = config.get('_precomputed_graph')
    if graph is None: graph = optimize_floor_graph(ids, edges, {uid: local_scale(r) for uid, r in rows.items()}, graph_config)
    elif {e['edge_id'] for e in graph['accepted_edges']}-{e['edge_id'] for e in edges}:
        raise ValueError('precomputed graph contains edges outside the inference policy')
    room_scores = config.get('room_relationships', [])
    rooms, openings, structures = [], [], {}
    member_room = {}
    for component in graph['components']:
        cid, members = component['component_id'], component['observation_ids']
        groups = complete_link_rooms(members, room_scores, config.get('room_threshold', .98)) if config.get('room_fusion', True) else [[x] for x in members]
        room_shapes = {}
        for group in groups:
            rid = 'room_'+fingerprint(group)[:20]; shapes = []
            for uid in group:
                member_room[uid] = rid
                p = observation_polygon(rows[uid])
                if p is not None: shapes.append(transform_shape(p, graph['poses'][uid]['T_local_to_component']))
            merged = unary_union(shapes) if shapes else None
            if merged is not None and merged.is_valid and not merged.is_empty: room_shapes[rid] = merged
            rooms.append({'room_id': rid, 'component_id': cid, 'observation_ids': group,
                          'geometry': geojson(merged), 'geometry_status': 'valid' if shapes else 'unavailable',
                          'fusion': 'union_fixed_prediction_regions',
                          'partition': 'tentative_space' if exploratory else 'predicted_complete_room',
                          'association_source': 'geometry_complete_link' if exploratory and len(group) > 1 else
                                                'single_observation' if exploratory else 'learned_or_single_observation'})
        structures[cid] = partition_graph(room_shapes, config.get('graph_snap_local', .03))
        for wall in structures[cid]['walls']:
            wall['observation_ids'] = sorted({uid for room in rooms if room['room_id'] in wall['room_ids'] for uid in room['observation_ids']})
        component['footprint'] = geojson(unary_union(list(room_shapes.values()))) if room_shapes else None
    for uid, row in rows.items():
        row.update(graph['poses'][uid]); row['predicted_room_id'] = member_room[uid]
        component = next(c for c in graph['components'] if c['component_id'] == row['component_id'])
        row['pose_status'] = 'isolated' if len(component['observation_ids']) == 1 else 'localized'
        p = observation_polygon(row)
        row['transformed_geometry'] = geojson(transform_shape(p, row['T_local_to_component'])) if p is not None else None
        for candidate in row.get('prediction', row).get('openings', []):
            value = dict(candidate, observation_id=uid, room_id=member_room[uid], component_id=row['component_id'])
            if candidate.get('endpoints_local') is not None:
                try: value['endpoints_component'] = points_transform(candidate['endpoints_local'], row['T_local_to_component']).tolist()
                except ValueError: value['endpoints_component'] = None
            else: value['endpoints_component'] = None
            # Wall identity is a derived attachment, never used to adjust an endpoint.
            value['structural_wall_ids'] = []
            if value['endpoints_component'] is not None:
                from portallayout.geometry.floor_geometry import point_segment_distances
                for wall in structures[row['component_id']]['walls']:
                    if member_room[uid] in wall['room_ids'] and point_segment_distances(value['endpoints_component'], [wall['endpoints']]).max() <= config.get('graph_snap_local', .03):
                        value['structural_wall_ids'].append(wall['wall_id'])
            openings.append(value)
    order = sorted(ids); index = {uid: i for i, uid in enumerate(order)}
    direct = np.zeros((len(ids), len(ids)), int); mask = np.zeros_like(direct, bool)
    reachable = np.zeros_like(direct, bool)
    room_edges = []
    accepted_ids = {e['edge_id'] for e in graph['accepted_edges']}
    for pair in pair_predictions:
        i, j = index[pair['observation_a']], index[pair['observation_b']]
        accepted = pair.get('match_status') == 'accepted' or (exploratory and pair.get('match_status') == 'tentative')
        if accepted or pair.get('match_status') in ('no_match', 'tentative_no_match'):
            mask[i, j] = mask[j, i] = True; direct[i, j] = direct[j, i] = int(accepted)
        if accepted:
            ra, rb = member_room[order[i]], member_room[order[j]]
            if ra != rb:
                # A shared visible opening alone does not prove opposite-side room adjacency.
                side_evidence = False
                if pair['pair_id'] in accepted_ids:
                    for match in pair.get('matches', []):
                        oa = next((o for o in openings if o['observation_id'] == order[i] and o['candidate_id'] == match['candidate_id_a']), None)
                        if oa and oa.get('endpoints_component'):
                            p, q = np.array(oa['endpoints_component']); d = q-p
                            ca = np.array(rows[order[i]]['T_local_to_component'])[:2, 2]
                            cb = np.array(rows[order[j]]['T_local_to_component'])[:2, 2]
                            side_evidence |= bool(np.cross(d, ca-p)*np.cross(d, cb-p) < -1e-8)
                room_edges.append({'room_a': ra, 'room_b': rb, 'pair_id': pair['pair_id'],
                    'status': 'portal_supported_opposite_side' if side_evidence else 'shared_opening_only',
                    'direct_adjacency': side_evidence})
    for c in graph['components']:
        ii = [index[x] for x in c['observation_ids']]; reachable[np.ix_(ii, ii)] = True
    return {'schema_version': 'floor-prediction.v2', 'floor_uid': config.get('floor_uid', 'floor'),
        'status': ('exploratory_' if exploratory else '')+('multi_observation_reconstruction' if any(len(c['observation_ids']) > 1 for c in graph['components']) else 'isolated_observations_only'),
        'inference_mode': mode, 'validation_status': 'uncalibrated_preliminary' if exploratory else 'strict_policy',
        'coordinate_spec': {'frame': 'bi_component_xz', 'units': config.get('units', 'camera_height_normalized'),
                            'scale_status': 'known' if config.get('units') == 'meters' else 'unknown',
                            'transform_direction': 'local_to_component'},
        'observation_order': order, 'observations': [rows[uid] for uid in order], 'rooms': rooms, 'openings': openings,
        'components': graph['components'], 'main_component_id': graph['main_component_id'], 'structural_graphs': structures,
        'pair_results': pair_predictions, 'pose_graph': graph,
        'adjacency': {'observation_order': order, 'shared_opening': direct.tolist(), 'evaluated_mask': mask.tolist(),
                      'reachable': reachable.tolist(), 'room_connections': room_edges,
                      'validation_status': 'tentative' if exploratory else 'strict_policy'},
        'room_relationships': room_scores, 'ground_truth_read': False,
        'geometry_policy': 'fixed_local_predictions; union_only_within_predicted_room'}


class FloorAssembler:
    def __init__(self, config=None): self.config = config or {}
    def __call__(self, observations, pair_predictions): return assemble_floor(observations, pair_predictions, self.config)
