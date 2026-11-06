"""Fixed-denominator, offline-only geometry intervention audit."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from portallayout.datasets.opening_pair_supervision import associate_candidates
from portallayout.datasets.zind_floor_dataset import read_jsonl
from portallayout.datasets.floor_auxiliary_supervision import pose_target
from portallayout.evaluation.floor_metrics import load_floor_ground_truth
from portallayout.evaluation.pose_candidate_audit import normalized_endpoints
from portallayout.geometry.pose_geometry import METHODS, opening_overlay, pose_hypotheses, regenerate_pair, variant_id
from portallayout.engine.pose_verifier_policy import canonical_candidates
from portallayout.utils.room_feature_cache import fingerprint, write_json_atomic

CONFIGS = [dict(pose_geometry=m, scale_shrinkage=s) for m in METHODS for s in (1., .5, 0.)]
BASELINE = variant_id(CONFIGS[0])


def relationship(a, b):
    return ('same_room' if a['room_id'] == b['room_id'] else 'cross_room') if a['membership_confirmed'] and b['membership_confirmed'] else 'unknown_room'


def truth_for(gt, a, b):
    ga, gb = gt['observations'][a], gt['observations'][b]
    return np.linalg.inv(ga['pose'])@np.asarray(gb['pose']) if ga['pose_valid'] and gb['pose_valid'] else None


def assess_endpoints(a, b, truth, alpha=1.):
    try: hs = pose_hypotheses(a, b, alpha)
    except (TypeError, ValueError): hs = []
    labels = [pose_target(h['T_B_to_A'], truth)[0] for h in hs]
    return dict(success=1 in labels, hypotheses=len(hs), labels=labels,
                errors=[pose_target(h['T_B_to_A'], truth)[1] for h in hs])


def interventions(pred, gt, depth):
    # Reuse the existing component interventions, but use the current Bi frame
    # and actual depth token count (the old oracle CLI assumes ZInD/256).
    from tools.validate_portal_geometry_hypothesis import edits
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    result = edits(pred, gt)
    u = (np.arctan2(gt[:, 0], gt[:, 1])/(2*np.pi)+.5) % 1
    at = u*len(depth)-.5; lo = np.floor(at).astype(int); fraction = at-lo
    radial = depth[lo % len(depth)]*(1-fraction)+depth[(lo+1) % len(depth)]*fraction
    result['O_angle'] = gt/np.linalg.norm(gt, axis=1)[:, None]*radial[:, None]
    return result


def audit_floor(cache, floor, raw, root, role, overlays):
    uid = floor['floor_uid']; base = Path(root)/'labels'/uid
    gt = load_floor_ground_truth(root, uid)
    annotations = {x['observation_id']: x for x in read_jsonl(base/'observations_gt.jsonl')}
    from portallayout.geometry.floor_geometry import points_transform
    projections, angles = [], []
    for oid, ann in annotations.items():
        if not gt['observations'][oid]['pose_valid']: continue
        for item in ann['interfaces_by_layout']['layout_raw']:
            if item.get('floor_endpoints') is None: continue
            local = normalized_endpoints(ann,item)
            projections.append(float(np.max(np.abs(points_transform(local,gt['observations'][oid]['pose'])-
                                                      np.asarray(item['floor_endpoints'])*[-1.,1.]))))
            uv=(item.get('projection') or {}).get('endpoint_u')
            if uv is not None:
                u=np.arctan2(local[:,0],local[:,1])/(2*np.pi)+.5
                angles.append(float(np.max(np.abs((u-np.asarray(uv)+.5)%1.-.5))))
    if max(projections,default=0.)>1e-6 or max(angles,default=0.)>1e-6:
        raise ValueError('real annotation coordinate projection failed')
    rows, oracle_rows, inventory = [], [], []
    for pair in raw:
        a, b = pair['observation_a'], pair['observation_b']
        meta = dict(pair_id=pair['pair_id'], floor_uid=uid, house_id=gt['house_id'], role=role,
                    relationship=relationship(gt['observations'][a], gt['observations'][b]),
                    gt_valid=truth_for(gt, a, b) is not None)
        inventory.append(dict(meta, status=pair['numerical_status']))
        seeds = [m for m in pair['matches'] if m['confidence'] >= .7]
        if not seeds: continue
        values = {}
        for config in CONFIGS:
            obs = overlays[config['pose_geometry']]
            candidate = regenerate_pair(pair, obs[a], obs[b], config)
            hs = [h for h in canonical_candidates(candidate) if h['match_confidence'] >= .7]
            labels = [pose_target(h['T_B_to_A'], truth_for(gt, a, b))[0] for h in hs]
            values[variant_id(config)] = dict(success=1 in labels, hypotheses=len(hs),
                labels=labels, failures=candidate['geometry_failures'])
        rows.append(dict(meta, collection='r3', seed_candidate_ids=[[m['candidate_id_a'], m['candidate_id_b']] for m in seeds], variants=values))
    associated, interfaces = {}, {}
    for obs_id, ann in annotations.items():
        if obs_id not in cache.records: continue
        mapping, items = associate_candidates(cache.records[obs_id], ann)
        for m in mapping:
            if m['status'] == 'associated':
                associated[m['interface_id']] = (obs_id, m['candidate_id'])
        interfaces.update(items)
    weak_groups = {}
    for link in read_jsonl(base/'portal_links.jsonl'):
        ia, ib = link['source_interface_a'], link['source_interface_b']
        if link.get('ambiguous') or ia not in associated or ib not in associated: continue
        a, ca = associated[ia]; b, cb = associated[ib]
        if a == b: continue
        if a > b: a, b, ca, cb, ia, ib = b, a, cb, ca, ib, ia
        truth = truth_for(gt, a, b)
        pair_id = fingerprint([a, b])
        meta = dict(pair_id=pair_id, floor_uid=uid, house_id=gt['house_id'], role=role,
                    relationship=relationship(gt['observations'][a], gt['observations'][b]), gt_valid=truth is not None)
        group = weak_groups.setdefault(pair_id, dict(meta, collection='weak', seed_candidate_ids=[],
            variants={variant_id(c): dict(success=False, hypotheses=0) for c in CONFIGS}))
        if [ca, cb] in group['seed_candidate_ids']: continue
        group['seed_candidate_ids'].append([ca, cb])
        for config in CONFIGS:
            obs = overlays[config['pose_geometry']]
            ea = next(o['endpoints_local'] for o in obs[a]['prediction']['openings'] if o['candidate_id'] == ca)
            eb = next(o['endpoints_local'] for o in obs[b]['prediction']['openings'] if o['candidate_id'] == cb)
            score = assess_endpoints(ea, eb, truth, config['scale_shrinkage'])
            value = group['variants'][variant_id(config)]
            value['success'] |= score['success']; value['hypotheses'] += score['hypotheses']
        ga = normalized_endpoints(annotations[a], interfaces[ia]); gb = normalized_endpoints(annotations[b], interfaces[ib])
        outputs = []
        for obs_id, cid, target in ((a, ca, ga), (b, cb, gb)):
            rec, arrays = cache.load(obs_id)
            e = next(o['endpoints_local'] for o in rec['prediction']['openings'] if o['candidate_id'] == cid)
            try: values = interventions(e, target, np.maximum(np.abs(arrays['depth_enclosed']), .001))
            except (ValueError, TypeError, IndexError): values = {}
            values['O_full'] = target
            outputs.append(values)
        conditions = {name: assess_endpoints(outputs[0].get(name), outputs[1].get(name), truth)
                      for name in ('O_angle','O_support_line','O_direction','O_midpoint','O_width','O_full')}
        oracle_rows.append(dict(meta, candidate_ids=[ca, cb], interface_ids=[ia, ib], conditions=conditions,
            association_or_annotation_inconsistent=truth is not None and not conditions['O_full']['success']))
    rows.extend(weak_groups.values())
    return dict(floor_uid=uid, role=role, inventory=inventory, rows=rows, oracles=oracle_rows,
                coordinate_audit=dict(checks=len(projections),projection_max_error=max(projections,default=0.),angle_max_error=max(angles,default=0.)),
                fixed_sample_fingerprint=fingerprint([(r['collection'], r['pair_id'], r['seed_candidate_ids']) for r in rows]))


def summarize(rows, key, scope, collection, role='selection'):
    selected = [r for r in rows if r['role'] == role and r['collection'] == collection and r['gt_valid'] and
                (scope is None or r['relationship'] == scope)]
    houses = defaultdict(list)
    for r in selected: houses[r['house_id']].append(bool(r['variants'][key]['success']))
    rates = {h: float(np.mean(v)) for h, v in houses.items()}
    count = sum(sum(v) for v in houses.values())
    return dict(pairs=len(selected), correct_pairs=int(count), success_rate=count/len(selected) if selected else None,
        house_macro=float(np.mean(list(rates.values()))) if rates else None, house_rates=rates,
        correct_houses=sum(any(v) for v in houses.values()), invalid_gt=sum(not r['gt_valid'] for r in rows
            if r['role'] == role and r['collection'] == collection and (scope is None or r['relationship'] == scope)))


def select_geometry(parts):
    rows = [r for p in parts for r in p['rows']]
    table = {}
    for config in CONFIGS:
        key = variant_id(config)
        table[key] = dict(config=config, statistics={role: {collection: {scope: summarize(rows,key,None if scope=='overall' else scope,collection,role)
            for scope in ('overall','cross_room','same_room','unknown_room')} for collection in ('r3','weak')} for role in ('fit','selection')})
    base = table[BASELINE]['statistics']['selection']; candidates = []
    for rank, (key, entry) in enumerate(table.items()):
        r = entry['statistics']['selection']['r3']['cross_room']; w = entry['statistics']['selection']['weak']['cross_room']
        br, bw = base['r3']['cross_room'], base['weak']['cross_room']
        available = all(x is not None for x in (r['house_macro'],br['house_macro'],w['success_rate'],bw['success_rate']))
        checks = dict(candidate_macro_improved=available and r['house_macro'] > br['house_macro'],
            correct_pairs_increased=r['correct_pairs'] > br['correct_pairs'],
            houses_not_decreased=r['correct_houses'] >= br['correct_houses'],
            weak_success_not_decreased=available and w['success_rate'] >= bw['success_rate'])
        entry['checks'] = checks; entry['passed'] = all(checks.values())
        houses = sorted(r['house_rates'])
        if houses:
            delta = np.array([r['house_rates'][h]-br['house_rates'][h] for h in houses])
            rng = np.random.RandomState(123)
            boot = delta[rng.randint(0,len(delta),size=(2000,len(delta)))].mean(1)
            entry['paired_house_delta_ci95'] = np.percentile(boot,[2.5,97.5]).tolist()
        if entry['passed']:
            total = entry['statistics']['selection']['r3']['overall']['correct_pairs']
            candidates.append(((r['house_macro'],w['success_rate'],total,-rank),key))
    winner = max(candidates)[1] if candidates else None
    return dict(status='geometry_gain_found' if winner else 'no_geometry_gain', selected=winner,
                selected_config=table[winner]['config'] if winner else None, baseline=BASELINE, variants=table,
                bootstrap=dict(unit='house',samples=2000,seed=123), calibration_used=False)
