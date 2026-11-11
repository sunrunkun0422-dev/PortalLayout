"""Offline auxiliary targets; original ZInD and visual-unknown labels stay intact."""
import json
import math
from pathlib import Path
import numpy as np
from portallayout.evaluation.floor_metrics import load_floor_ground_truth
from portallayout.geometry.floor_geometry import similarity, parameters
from portallayout.utils.room_feature_cache import fingerprint, content_hash, write_json_atomic
from portallayout.models.floor_auxiliary import FEATURE_VERSION


def house_protocol(dataset_root, seed=123, protocol_path=None):
    if protocol_path is not None:
        path = Path(protocol_path).resolve()
        result = json.loads(path.read_text())
        claimed = result.get('fingerprint')
        if not isinstance(claimed, str) or fingerprint({k: v for k, v in result.items() if k != 'fingerprint'}) != claimed:
            raise ValueError('house protocol fingerprint mismatch')
        roles = ('fit', 'selection', 'calibration', 'audit')
        if result.get('schema_version') != 'research10.house-protocol.v1':
            raise ValueError('unsupported custom house protocol schema')
        groups = []
        for role in roles:
            houses = result.get(role)
            if not isinstance(houses, list) or not houses or any(not isinstance(h, str) or not h for h in houses):
                raise ValueError('custom house protocol requires a nonempty list for ' + role)
            if len(houses) != len(set(houses)): raise ValueError('duplicate house in ' + role)
            groups.append(set(houses))
        if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
            raise ValueError('house split overlap')
        official = json.loads((Path(dataset_root)/'splits/official_house_split.json').read_text())
        if not set.union(*groups) <= set(official['train']):
            raise ValueError('custom house protocol includes a house outside official train')
        cohort_path = path.parent/'cohort.json'
        if not cohort_path.is_file() or content_hash(cohort_path) != result.get('source_sha256'):
            raise ValueError('house protocol cohort source mismatch')
        cohort = json.loads(cohort_path.read_text())
        if any(cohort.get('roles', {}).get(role) != result[role] for role in roles):
            raise ValueError('house protocol differs from source cohort')
        return result
    path = Path(dataset_root)/'splits/derived_house_split.json'
    split = json.loads(path.read_text())
    dev = sorted(split['dev'], key=lambda h: (fingerprint([seed, h]), h))
    middle = len(dev)//2
    result = {'seed': seed, 'fit': sorted(split['fit']), 'selection': sorted(dev[:middle]),
              'calibration': sorted(dev[middle:]), 'source_sha256': content_hash(path)}
    groups = [set(result[k]) for k in ('fit', 'selection', 'calibration')]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]): raise ValueError('house split overlap')
    result['fingerprint'] = fingerprint(result)
    return result


def house_role(house, protocol):
    for role in ('fit', 'selection', 'calibration', 'audit'):
        if house in protocol.get(role, ()): return role
    return 'heldout'


def validate_cache_house_roles(cache, dataset_root, protocol, allowed_roles, cohort_path=None):
    """Check every cached floor against its authoritative public dataset index."""
    from portallayout.datasets.zind_floor_dataset import read_jsonl, safe_path, validate_public_manifest
    samples = list(read_jsonl(Path(dataset_root)/'samples/all_panoramas.jsonl'))
    by_id = {row['sample_id']: row for row in samples}
    if len(by_id) != len(samples): raise ValueError('duplicate dataset sample identity')
    cohort_floors = None
    if cohort_path is not None:
        cohort = json.loads(Path(cohort_path).read_text())
        rows = cohort.get('cohort', [])
        cohort_floors = {row['floor_uid']: (row['house_id'], row['role']) for row in rows}
        if not rows or len(cohort_floors) != len(rows): raise ValueError('invalid cohort floor identities')
    result = {}
    for floor in cache.floors:
        sample = by_id.get(floor['sample_id'])
        if (sample is None or sample['floor_uid'] != floor['floor_uid'] or
                sample['official_split'] != 'train'):
            raise ValueError('cached floor is not an official training sample')
        manifest = validate_public_manifest(json.loads(safe_path(Path(dataset_root), sample['input_manifest']).read_text()))
        if (manifest['sample_id'] != floor['sample_id'] or manifest['floor_uid'] != floor['floor_uid'] or
                manifest['observation_order'] != floor['observation_order']):
            raise ValueError('cached floor differs from authoritative RGB input manifest')
        role = house_role(sample['house_id'], protocol)
        if role not in allowed_roles:
            raise ValueError('cache contains forbidden {} house {}'.format(role, sample['house_id']))
        if cohort_floors is not None and cohort_floors.get(floor['floor_uid']) != (sample['house_id'], role):
            raise ValueError('cached floor is not the frozen cohort floor')
        if floor['floor_uid'] in result: raise ValueError('duplicate cached floor identity')
        result[floor['floor_uid']] = sample['house_id']
    if not result: raise ValueError('empty cache')
    return result


def validate_supervision_house_roles(data, protocol, allowed_roles, floor_houses=None, cache=None):
    """Fail closed on audit/heldout rows, including rows otherwise skipped by training."""
    floor_observations = ({floor['floor_uid']: set(floor['observation_order']) for floor in cache.floors}
                          if cache is not None else None)
    for row in data.get('pairs', []) + data.get('records', []) + data.get('opening_pairs', []):
        house, role = row['house_id'], house_role(row['house_id'], protocol)
        if role not in allowed_roles:
            raise ValueError('supervision contains forbidden {} house {}'.format(role, house))
        if 'role' in row and row['role'] != role: raise ValueError('supervision house role mismatch')
        if floor_houses is not None and floor_houses.get(row['floor_uid']) != house:
            raise ValueError('supervision floor/house disagrees with prediction cache')
        if floor_observations is not None and any(row.get('observation_'+side) not in floor_observations[row['floor_uid']]
                                                   for side in ('a', 'b') if 'observation_'+side in row):
            raise ValueError('supervision observation disagrees with prediction floor')


def validate_auxiliary_for_training(path, protocol):
    data = json.loads(Path(path).read_text())
    if data.get('house_protocol') != protocol:
        raise ValueError('auxiliary supervision and requested house protocols differ')
    validate_supervision_house_roles(data, protocol, {'fit', 'selection', 'calibration'})
    return data


def pose_target(predicted, truth):
    try:
        p, g = similarity(predicted), similarity(truth)
        forward = float(np.linalg.norm(p[:2, 2]-g[:2, 2]))
        reverse = float(np.linalg.norm(np.linalg.inv(p)[:2, 2]-np.linalg.inv(g)[:2, 2]))
        yaw = parameters(p)[2]-parameters(g)[2]
        angle = abs(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))))
        ratio = math.exp(parameters(p)[3]-parameters(g)[3])
        scale_error = max(abs(ratio-1.), abs(1./ratio-1.))
        positive = max(forward, reverse) <= .25 and angle <= 10. and scale_error <= .1
        negative = max(forward, reverse) >= .5 or angle >= 20. or scale_error >= .2
        return (1 if positive else 0 if negative else -1), {'forward_translation_height_units': forward,
            'reverse_translation_height_units': reverse, 'rotation_degrees': angle, 'relative_scale_error': scale_error}
    except (ValueError, TypeError, np.linalg.LinAlgError): return -1, {'reason': 'pose_gt_unavailable_or_invalid'}


def build_auxiliary_supervision(cache, dataset_root, pair_checkpoint, output, device='cpu', seed=123, protocol=None):
    from portallayout.engine.floor_layout_pipeline import FloorModelBundle, encode_floor, predict_pair
    from portallayout.engine.opening_pair_training import read_supervision
    from portallayout.datasets.opening_pair_supervision import build_weak_supervision
    output = Path(output).resolve(); dataset_root = Path(dataset_root).resolve()
    if (output == cache.root or cache.root in output.parents or output == dataset_root or dataset_root in output.parents
            or (output.exists() and any(output.iterdir()))): raise ValueError('use a new auxiliary supervision directory')
    protocol = protocol or house_protocol(dataset_root, seed)
    if 'audit' in protocol:
        validate_cache_house_roles(cache, dataset_root, protocol, {'fit', 'selection', 'calibration'})
    output.mkdir(parents=True, exist_ok=True); (output/'features').mkdir()
    weak = build_weak_supervision(cache, dataset_root, output/'opening_weak.json', protocol=protocol if 'audit' in protocol else None)
    weak_rows = {(r['observation_a'], r['observation_b']): r for r in weak['pairs']}
    bundle = FloorModelBundle({'pair_checkpoint': str(Path(pair_checkpoint).resolve())}, device)
    records, pair_records, annotation_hashes = [], [], {}
    import itertools
    for floor in cache.floors:
        gt = load_floor_ground_truth(dataset_root, floor['floor_uid'])
        for p in gt['source_files']: annotation_hashes[p] = content_hash(p)
        role = house_role(gt['house_id'], protocol)
        if role not in ('fit', 'selection', 'calibration'): raise ValueError('audit/heldout floors must not enter training/calibration supervision export')
        observations, views, _ = encode_floor(cache, floor['observation_order'], bundle)
        lookup = {o['observation_id']: o for o in observations}
        for aid, bid in itertools.combinations(sorted(floor['observation_order']), 2):
            if aid not in views or bid not in views: continue
            ga, gb = gt['observations'][aid], gt['observations'][bid]
            truth = np.linalg.inv(ga['pose']) @ np.asarray(gb['pose']) if ga['pose_valid'] and gb['pose_valid'] else None
            def save_feature(task, pair, hypothesis, features):
                label, error = pose_target(hypothesis['T_B_to_A'], truth)
                room_valid = label == 1 and ga['membership_confirmed'] and gb['membership_confirmed']
                target = label if task == 'pose' else int(ga['room_id'] == gb['room_id']) if room_valid else -1
                uid = fingerprint([task, hypothesis['hypothesis_id']])
                path = output/'features'/(uid+'.npz')
                np.savez_compressed(path, raster=features['raster'].astype(np.uint8), visual=features['visual'].astype(np.float16),
                                    geometry=features['geometry'].astype(np.float32))
                records.append({'record_id': uid, 'task': task, 'pair_id': pair['pair_id'],
                    'hypothesis_id': hypothesis['hypothesis_id'], 'floor_uid': floor['floor_uid'],
                    'observation_a': aid, 'observation_b': bid, 'house_id': gt['house_id'], 'role': role,
                    'candidate_ids': hypothesis['seed_candidate_ids'], 'T_B_to_A': hypothesis['T_B_to_A'],
                    'opening_score': next(m['confidence'] for m in pair['matches']
                        if [m['candidate_id_a'], m['candidate_id_b']] == hypothesis['seed_candidate_ids']),
                    'target': target, 'pose_error_offline': error, 'membership_valid': room_valid,
                    'independent_evidence': features['independent_evidence'], 'feature_file': str(path.relative_to(output)),
                    'feature_sha256': content_hash(path), 'source': 'audited_pose_geometry' if task == 'pose' else 'confirmed_complete_room',
                    'source_cache_keys': [cache.records[aid]['cache_key'], cache.records[bid]['cache_key']]})
            pair = predict_pair(lookup[aid], lookup[bid], views[aid], views[bid], bundle, feature_callback=save_feature)
            weak_row = weak_rows.get((aid, bid))
            if weak_row:
                cells = np.array(weak_row['weak_cells'], int).reshape(len(weak_row['candidate_ids_a']), len(weak_row['candidate_ids_b']))
                matches = [dict(m, target=int(cells[m['index_a'], m['index_b']])) for m in pair['matches']]
                pair_records.append({'pair_id': pair['pair_id'], 'house_id': gt['house_id'], 'role': role,
                    'matches': matches, 'known_negative_pair': bool(cells.size and (cells == 0).all()),
                    'numerical_status': pair.get('numerical_status'), 'floor_uid': floor['floor_uid']})
        print('auxiliary export floor={} records={}'.format(floor['floor_uid'], len(records)), flush=True)
    if any(content_hash(p) != h for p, h in annotation_hashes.items()): raise ValueError('GT changed during target export')
    stats = {task: {role: {str(label): sum(r['task'] == task and r['role'] == role and r['target'] == label for r in records)
                          for label in (-1, 0, 1)} for role in ('fit', 'selection', 'calibration')} for task in ('pose', 'room')}
    payload = {'schema_version': 'floor-auxiliary-supervision.v1', 'feature_version': FEATURE_VERSION,
        'pair_checkpoint_sha256': content_hash(pair_checkpoint), 'source_snapshot': cache.snapshot,
        'source_contract': cache.fingerprint, 'house_protocol': protocol, 'annotation_sha256': annotation_hashes,
        'source_cache_files_sha256': cache.before, 'records': records, 'opening_pairs': pair_records,
        'statistics': stats, 'original_visual_labels_modified': False, 'original_pose_training_mask_modified': False}
    payload['fingerprint'] = fingerprint(payload)
    cache.check_unchanged(); write_json_atomic(output/'supervision.json', payload)
    return payload
