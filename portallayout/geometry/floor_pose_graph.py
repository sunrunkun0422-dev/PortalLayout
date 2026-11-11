"""Deterministic, component-wise Sim(2) forests and robust pose optimization."""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.optimize import least_squares
from portallayout.geometry.floor_geometry import similarity, parameters, matrix, sim2_log


@dataclass(frozen=True)
class PoseGraphConfig:
    mode: str = 'robust'
    translation_sigma: float = .05
    rotation_sigma_degrees: float = 5.
    log_scale_sigma: float = .05
    conflict_sigma: float = 3.
    rounds: int = 3
    max_nfev: int = 200
    sparse_jacobian: bool = False


def components(nodes, edges):
    graph = {n: set() for n in nodes}
    for e in edges:
        graph[e['a']].add(e['b']); graph[e['b']].add(e['a'])
    groups, seen = [], set()
    for uid in sorted(graph):
        if uid in seen: continue
        group, pending = [], [uid]; seen.add(uid)
        while pending:
            n = pending.pop(); group.append(n)
            for v in sorted(graph[n]-seen): seen.add(v); pending.append(v)
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g))


def forest(nodes, edges):
    parent = {x: x for x in nodes}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    selected = []
    for e in sorted(edges, key=lambda e: (-e['weight'], e['edge_id'])):
        a, b = root(e['a']), root(e['b'])
        if a != b:
            parent[b] = a; selected.append(e)
    return selected


def path_transform(edges, source, target, excluded=()):
    """Return T_target_to_source along a stable shortest path excluding edges."""
    graph = {}
    for e in sorted(edges, key=lambda e: e['edge_id']):
        if e['edge_id'] in excluded: continue
        m = similarity(e['T_B_to_A'])
        graph.setdefault(e['a'], []).append((e['b'], m, e['edge_id']))
        graph.setdefault(e['b'], []).append((e['a'], np.linalg.inv(m), e['edge_id']))
    pending, seen = [(source, np.eye(3), [])], {source}
    for node, accumulated, path in pending:
        if node == target: return accumulated, path
        for neighbor, relative, eid in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor); pending.append((neighbor, accumulated @ relative, path+[eid]))
    return None, []


def initialize(nodes, tree):
    anchor = min(nodes)
    poses = {}
    for uid in nodes:
        pose, _ = path_transform(tree, anchor, uid)
        if pose is None: raise ValueError('forest does not span component')
        poses[uid] = pose
    return poses


def edge_residual(e, poses, scales, config):
    measured = similarity(e['T_B_to_A'])
    estimate = np.linalg.inv(poses[e['a']]) @ poses[e['b']]
    # Error lives in B's local coordinate system; normalize by B's predicted scale.
    error = sim2_log(np.linalg.inv(measured) @ estimate)
    sigma = [config.translation_sigma*scales.get(e['b'], 1.)]*2 + [
        math.radians(config.rotation_sigma_degrees), config.log_scale_sigma]
    return error / np.asarray(sigma)


def _huber_cost(residual):
    a = np.abs(residual)
    return float(np.where(a <= 1., .5*a*a, a-.5).sum())


def _sim2_logs(matrices):
    """Vector form of sim2_log for already validated similarities in the solver."""
    angle = np.arctan2(matrices[:,1,0], matrices[:,0,0])
    scale = .5*np.log(np.linalg.det(matrices[:,:2,:2]))
    z = scale+1j*angle; coefficient = np.empty_like(z)
    small = np.abs(z) < 1e-7
    coefficient[small] = 1.+z[small]/2.+z[small]**2/6.
    coefficient[~small] = np.expm1(z[~small])/z[~small]
    translation = (matrices[:,0,2]+1j*matrices[:,1,2])/coefficient
    return np.stack([translation.real,translation.imag,angle,scale],axis=1)


def solve_component(nodes, edges, scales, config):
    tree = forest(nodes, edges)
    initial = initialize(nodes, tree)
    result = dict(poses=initial, initial=initial, tree=tree, optimization_status='forest_only', history=[], rejected=[])
    if config.mode == 'none' or len(nodes) == 1:
        return result
    anchor, free = min(nodes), sorted(set(nodes)-{min(nodes)})
    x = np.concatenate([parameters(initial[n]) for n in free])
    weights = {e['edge_id']: e['weight'] for e in edges}
    active, persistence = list(edges), {}
    def unpack(values):
        return dict({anchor: np.eye(3)}, **{n: matrix(values[i*4:(i+1)*4]) for i, n in enumerate(free)})
    try:
        for iteration in range(config.rounds):
            if config.sparse_jacobian:
                order = sorted(nodes); index = {uid: i for i,uid in enumerate(order)}
                aa = np.array([index[e['a']] for e in active]); bb = np.array([index[e['b']] for e in active])
                inverse_measured = np.linalg.inv(np.asarray([e['T_B_to_A'] for e in active]))
                normalizers = np.array([[config.translation_sigma*scales.get(e['b'],1.)]*2+
                    [math.radians(config.rotation_sigma_degrees),config.log_scale_sigma] for e in active])
                root_weights = np.sqrt([weights[e['edge_id']] for e in active])[:,None]
            def residual(values):
                poses = unpack(values)
                if config.sparse_jacobian:
                    stacked = np.stack([poses[uid] for uid in order])
                    error = inverse_measured @ np.linalg.inv(stacked)[aa] @ stacked[bb]
                    return (_sim2_logs(error)/normalizers*root_weights).ravel()
                return np.concatenate([math.sqrt(weights[e['edge_id']])*edge_residual(e, poses, scales, config)
                                       for e in active])
            before = _huber_cost(residual(x))
            lower, upper = np.full_like(x, -np.inf), np.full_like(x, np.inf)
            lower[3::4], upper[3::4] = -math.log(100.), math.log(100.)
            kwargs = {}
            if config.sparse_jacobian:
                from scipy.sparse import lil_matrix
                pattern = lil_matrix((4*len(active), len(x)), dtype=int)
                indices = {uid: i for i, uid in enumerate(free)}
                for i, edge in enumerate(active):
                    for uid in (edge['a'], edge['b']):
                        if uid in indices:
                            j = indices[uid]; pattern[4*i:4*i+4, 4*j:4*j+4] = 1
                kwargs['jac_sparsity'] = pattern.tocsr()
            fit = least_squares(residual, x, loss='huber', f_scale=1., bounds=(lower, upper), max_nfev=config.max_nfev, **kwargs)
            after = _huber_cost(residual(fit.x))
            if not fit.success or not np.isfinite(fit.x).all() or after > before+1e-7:
                raise ValueError('optimizer failed to converge or increased robust cost')
            x, current = fit.x, unpack(fit.x)
            result['history'].append({'round': iteration+1, 'initial_cost': before, 'final_cost': after,
                                      'nfev': fit.nfev})
            rejected = []
            for e in active:
                eid = e['edge_id']; error = float(np.linalg.norm(edge_residual(e, current, scales, config)))
                persistence[eid] = persistence.get(eid, 0)+1 if error > config.conflict_sigma else 0
                weights[eid] = e['weight']*min(1., config.conflict_sigma/max(error, 1e-9))
                # Two edge-disjoint independent paths must agree against this measurement.
                first, p1 = path_transform(active, e['a'], e['b'], [eid])
                second, p2 = path_transform(active, e['a'], e['b'], [eid]+p1)
                if first is None or second is None or persistence[eid] < 2: continue
                hypothetical = {e['a']: np.eye(3), e['b']: first}
                other = dict(e, T_B_to_A=second.tolist())
                paths_agree = np.linalg.norm(edge_residual(other, hypothetical, scales, config)) <= config.conflict_sigma
                path_disagrees = np.linalg.norm(edge_residual(e, hypothetical, scales, config)) > config.conflict_sigma
                path_quality = min(next(v['weight'] for v in active if v['edge_id'] == k) for k in p1+p2)
                if paths_agree and path_disagrees and path_quality >= e['weight']:
                    rejected.append(eid)
            result['rejected'].extend(rejected)
            if rejected:
                active = [e for e in active if e['edge_id'] not in rejected]
                # Caller rebuilds components and anchors after pruning, never keeps stale poses.
                break
        result.update(poses=unpack(x), optimization_status='optimized', final_weights=weights)
    except (ValueError, np.linalg.LinAlgError, OverflowError) as exc:
        result.update(poses=initial, optimization_status='optimizer_failed', failure_reason=str(exc), rejected=[])
    return result


def optimize_floor_graph(observation_ids, edges, scales=None, config=None):
    config = config or PoseGraphConfig()
    if config.mode not in ('none', 'robust') or config.rounds not in (1, 2, 3):
        raise ValueError('invalid pose graph configuration')
    if min(config.translation_sigma, config.rotation_sigma_degrees, config.log_scale_sigma, config.conflict_sigma) <= 0:
        raise ValueError('pose graph tolerances must be positive')
    nodes = sorted(observation_ids)
    if len(nodes) != len(set(nodes)): raise ValueError('duplicate observation ID')
    scales = scales or {}
    if any(not np.isfinite(v) or v <= 0 for v in scales.values()): raise ValueError('invalid predicted local scale')
    valid, invalid, edge_ids = [], [], set()
    for value in edges:
        e = dict(value)
        try:
            if e['a'] == e['b'] or e['a'] not in nodes or e['b'] not in nodes: raise ValueError('invalid edge endpoints')
            if e['edge_id'] in edge_ids: raise ValueError('duplicate edge ID')
            edge_ids.add(e['edge_id'])
            e['T_B_to_A'] = similarity(e['T_B_to_A']).tolist()
            if not np.isfinite(e['weight']) or not 0 < e['weight'] <= 1: raise ValueError('invalid edge weight')
            valid.append(e)
        except (KeyError, ValueError) as exc:
            invalid.append({'edge': e, 'reason': str(exc)})
    rejected, solutions = [], []
    remaining_rounds = config.rounds
    optimization_passes = []
    # Rebuilding after pruning shares the original optimization budget; it
    # must not restart three rounds on each newly formed component.
    for rebuild in range(config.rounds):
        solutions, newly_rejected = [], []
        for members in components(nodes, valid):
            local = [e for e in valid if e['a'] in members and e['b'] in members]
            solved = solve_component(members, local, scales, replace(config, rounds=remaining_rounds))
            newly_rejected.extend(solved['rejected'])
            solutions.append((members, local, solved))
        if not newly_rejected: break
        spent = max((len(solved['history']) for _, _, solved in solutions), default=0)
        optimization_passes.append({'rebuild': rebuild, 'rounds': spent, 'rejected_edges': list(newly_rejected)})
        remaining_rounds -= spent
        rejected.extend(newly_rejected)
        valid = [e for e in valid if e['edge_id'] not in newly_rejected]
        if remaining_rounds <= 0: break
    if newly_rejected:
        solutions = [(g, [e for e in valid if e['a'] in g and e['b'] in g],
                      solve_component(g, [e for e in valid if e['a'] in g and e['b'] in g], scales,
                                      PoseGraphConfig(mode='none')))
                     for g in components(nodes, valid)]
    output, all_poses = [], {}
    for members, local, solved in solutions:
        cid = 'component_'+min(members)
        diagnostics = []
        for e in local:
            alt, path = path_transform(local, e['a'], e['b'], [e['edge_id']])
            residual = float(np.linalg.norm(edge_residual(e, solved['poses'], scales, config)))
            diagnostics.append({'edge_id': e['edge_id'], 'residual_sigma': residual,
                'weight': solved.get('final_weights', {}).get(e['edge_id'], e['weight']),
                'bridge': alt is None, 'independent_path': path,
                'status': 'cycle_conflict' if residual > config.conflict_sigma else 'accepted'})
        poses = {uid: solved['poses'][uid].tolist() for uid in members}
        all_poses.update({uid: {'component_id': cid, 'T_local_to_component': poses[uid]} for uid in members})
        output.append({'component_id': cid, 'observation_ids': members, 'anchor_id': min(members),
            'poses': poses, 'initial_poses': {uid: solved['initial'][uid].tolist() for uid in members},
            'tree_edge_ids': [e['edge_id'] for e in solved['tree']], 'edges': diagnostics,
            'edge_quality_sum': sum(e['weight'] for e in local), 'optimization_status': solved['optimization_status'],
            'optimization_history': solved['history'], 'failure_reason': solved.get('failure_reason'),
            'relative_position_to_other_components': 'unknown'})
    output.sort(key=lambda c: (-len(c['observation_ids']), -c['edge_quality_sum'], c['anchor_id']))
    return {'components': output, 'poses': all_poses, 'accepted_edges': valid,
            'main_component_id': output[0]['component_id'] if output else None,
            'pruning_passes': optimization_passes,
            'rejected_edges': sorted(set(rejected)), 'invalid_edges': invalid}


class FloorPoseGraph:
    def __init__(self, config=None): self.config = config or PoseGraphConfig()
    def __call__(self, observation_ids, edges, scales=None):
        return optimize_floor_graph(observation_ids, edges, scales, self.config)
