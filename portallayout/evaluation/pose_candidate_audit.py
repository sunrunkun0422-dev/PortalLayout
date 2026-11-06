"""Offline coordinate and four-stage pose ceiling audit; never inference input."""
import math
from collections import defaultdict
import numpy as np
from portallayout.datasets.floor_auxiliary_supervision import pose_target
from portallayout.datasets.opening_pair_supervision import associate_candidates
from portallayout.geometry.floor_geometry import matrix, points_transform
from portallayout.geometry.joint_layout import generate_opening_pose_hypotheses
from portallayout.engine.opening_pair_pipeline import connected_components


def synthetic_coordinate_audit():
    b = np.array([[.3, 1.], [1.2, 1.]])
    tests = []
    for yaw in (0., math.pi/2, -math.pi/2, math.pi-1e-8):
        for scale in (.5, 1., 2.):
            t = matrix([1.3, -.7, yaw, math.log(scale)])
            a = points_transform(b, t)
            for flip in (False, True):
                candidates = generate_opening_pose_hypotheses(a, b[::-1] if flip else b)['hypotheses']
                error = min(float(np.max(np.abs(np.asarray(h['T_B_to_A'])-t))) for h in candidates)
                assert error < 1e-8, (yaw, scale, flip, error)
                assert np.allclose(points_transform(a, np.linalg.inv(t)), b)
                tests.append(dict(yaw=yaw, scale=scale, flipped=flip, maximum_matrix_error=error))
    return dict(status='passed', cases=tests)


def normalized_endpoints(annotation, item):
    height = 1. if annotation['coordinate_spec'].get('camera_height_normalized_to_one_by_source') else annotation['camera_height_annotation']
    if height is None or height <= 0: raise ValueError('missing positive camera height')
    return np.asarray(item['local_endpoints'], float)*[-1., 1.]/height


def audit_floor(gt, annotations, portal_links, records, observations, raw_pairs, selected_pairs):
    projection_errors, angle_errors = [], []
    for uid, annotation in annotations.items():
        if not gt['observations'][uid]['pose_valid']: continue
        for item in annotation['interfaces_by_layout']['layout_raw']:
            if item.get('floor_endpoints') is None: continue
            local = normalized_endpoints(annotation, item)
            transformed = points_transform(local, np.asarray(gt['observations'][uid]['pose']))
            projection_errors.append(float(np.max(np.abs(transformed-np.asarray(item['floor_endpoints'])*[-1.,1.]))))
            projection = item.get('projection') or {}
            if projection.get('endpoint_u') is not None:
                u = np.arctan2(local[:,0],local[:,1])/(2*np.pi)+.5
                delta = (u-np.asarray(projection['endpoint_u'])+.5)%1.-.5
                angle_errors.append(float(np.max(np.abs(delta))))
    projection_max = max(projection_errors, default=0.); angle_max=max(angle_errors, default=0.)
    if projection_max > 1e-6 or angle_max > 1e-6: raise ValueError('real annotation coordinate audit failed: {}'.format((projection_max, angle_max)))
    interfaces = {}; associated = defaultdict(dict)
    for uid, ann in annotations.items():
        interfaces.update({x['interface_observation_id']: (uid,x) for x in ann['interfaces_by_layout']['layout_raw']})
        if uid in records:
            mapping, _ = associate_candidates(records[uid], ann)
            pred = {x['candidate_id']: x for x in records[uid]['prediction']['openings']}
            for m in mapping:
                if m['status']=='associated' and pred[m['candidate_id']].get('geometry_status')=='mapped_dominant_wall':
                    associated[uid][m['interface_id']] = pred[m['candidate_id']]['endpoints_local']
    stages = {s: defaultdict(list) for s in ('gt_geometry','predicted_geometry_known_correspondence','r3_candidates','selected_geometry')}
    def add(stage, aid, bid, aa, bb):
        if not (gt['observations'][aid]['pose_valid'] and gt['observations'][bid]['pose_valid']): return
        truth = np.linalg.inv(gt['observations'][aid]['pose'])@np.asarray(gt['observations'][bid]['pose'])
        try: hs=generate_opening_pose_hypotheses(aa,bb)['hypotheses']
        except ValueError:return
        stages[stage][tuple(sorted((aid,bid)))].extend(pose_target(h['T_B_to_A'],truth)[0] for h in hs)
    for link in portal_links:
        if link.get('ambiguous'):continue
        ia,ib=link['source_interface_a'],link['source_interface_b']
        if ia not in interfaces or ib not in interfaces:continue
        aid,aa=interfaces[ia];bid,bb=interfaces[ib]
        if aid not in gt['observations'] or bid not in gt['observations']:continue
        add('gt_geometry',aid,bid,normalized_endpoints(annotations[aid],aa),normalized_endpoints(annotations[bid],bb))
        if ia in associated[aid] and ib in associated[bid]:
            add('predicted_geometry_known_correspondence',aid,bid,associated[aid][ia],associated[bid][ib])
    selected={p['pair_id']:p for p in selected_pairs}
    for pair in raw_pairs:
        aid,bid=pair['observation_a'],pair['observation_b']; ga,gb=gt['observations'][aid],gt['observations'][bid]
        if not ga['pose_valid'] or not gb['pose_valid']:continue
        truth=np.linalg.inv(ga['pose'])@np.asarray(gb['pose'])
        from portallayout.engine.pose_verifier_policy import canonical_candidates
        hs=[h for h in canonical_candidates(pair) if h['match_confidence']>=.7]
        stages['r3_candidates'][(aid,bid)].extend(pose_target(h['T_B_to_A'],truth)[0] for h in hs)
        p=selected.get(pair['pair_id'],{})
        if p.get('T_B_to_A') is not None:stages['selected_geometry'][(aid,bid)].append(pose_target(p['T_B_to_A'],truth)[0])
    report={}
    for name,pairs in stages.items():
        strata={};edges=[]
        for scope in ('same_room','cross_room','unknown_room'):
            groups=[]
            for (a,b),labels in pairs.items():
                if not labels:continue
                ga,gb=gt['observations'][a],gt['observations'][b]
                relation=('same_room' if ga['room_id']==gb['room_id'] else 'cross_room') if ga['membership_confirmed'] and gb['membership_confirmed'] else 'unknown_room'
                if relation==scope:
                    groups.append(labels)
                    if 1 in labels:edges.append((a,b))
            strata[scope]=dict(pairs=len(groups),positive_pairs=sum(1 in x for x in groups),hypotheses=sum(len(x) for x in groups),positive_hypotheses=sum(x.count(1) for x in groups))
        report[name]=dict(strata=strata,oracle_component_sizes=[len(x) for x in connected_components(list(gt['observations']),edges)])
    return dict(floor_uid=gt['floor_uid'],house_id=gt['house_id'],coordinate_projection_max_error=projection_max,
                panorama_projection_max_error=angle_max,real_projection_checks=len(projection_errors),stages=report,
                gt_oracle_offline_only=True)
