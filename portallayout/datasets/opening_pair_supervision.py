"""Offline geometry-weak targets. Never imported by prediction-only inference."""
import itertools
import json
from pathlib import Path

import numpy as np

from portallayout.datasets.zind_floor_dataset import read_jsonl, iter_floor_pairs, safe_path
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def circular_iou(start_a, length_a, start_b, length_b):
    def pieces(start, length):
        if not np.isfinite([start, length]).all() or not 0 < length <= 1:
            return []
        start %= 1
        return [(start, min(1., start+length))] + ([(0., start+length-1)] if start+length > 1 else [])
    a, b = pieces(start_a, length_a), pieces(start_b, length_b)
    intersection = sum(max(0., min(y, v)-max(x, u)) for x, y in a for u, v in b)
    union = length_a+length_b-intersection
    return intersection/union if a and b and union > 0 else 0.


def associate_candidates(record, annotation, iou_threshold=.5, margin=.1):
    openings = record['prediction']['openings']
    n = record['tensor_spec']['depth_enclosed']['shape'][0]
    gt = [x for x in annotation['interfaces_by_layout']['layout_raw'] if x['kind'] in ('door', 'opening')]
    matrix = np.zeros((len(openings), len(gt)))
    coverage = np.zeros_like(matrix)
    for i, opening in enumerate(openings):
        left, right = opening['token_interval_inclusive']
        length = ((right-left) % n+1)/n
        for j, item in enumerate(gt):
            p = item.get('projection') or {}
            gl = p.get('length_u', 0.)
            if gl <= 0:
                continue
            score = circular_iou(left/n, length, p['start_u'], gl)
            matrix[i, j] = score
            # Intersection from IoU; detect one prediction covering multiple GTs.
            coverage[i, j] = score*(length+gl)/(1+score)/gl
    result = []
    inventory = annotation.get('layout_validity', {}).get('layout_raw_interface_inventory', False)
    for i, opening in enumerate(openings):
        item = {'candidate_id': opening['candidate_id'], 'interface_id': None,
                'status': 'ambiguous', 'iou': 0., 'source': 'geometry_weak'}
        if not inventory or not len(gt):
            item['status'] = 'missing_annotation'; result.append(item); continue
        j = int(matrix[i].argmax()); score = float(matrix[i, j])
        runner = np.sort(matrix[i])[-2] if len(gt) > 1 else 0.
        reverse_runner = np.sort(matrix[:, j])[-2] if len(openings) > 1 else 0.
        unique = (score >= iou_threshold and score-runner >= margin and
                  int(matrix[:, j].argmax()) == i and score-reverse_runner >= margin and
                  int((matrix[:, j] >= iou_threshold).sum()) == 1 and
                  int((coverage[i] >= .5).sum()) == 1)
        item.update(iou=score, best_interface_id=gt[j]['interface_observation_id'])
        if unique:
            item.update(interface_id=gt[j]['interface_observation_id'], status='associated')
        result.append(item)
    return result, {x['interface_observation_id']: x for x in gt}


def complete_targets(cells):
    """-1 unknown cells; output -2 ignored rows, -1 proven unmatched rows."""
    matrix = np.asarray(cells, dtype=np.int64)
    if matrix.ndim != 2 or not np.isin(matrix, [-1, 0, 1]).all():
        raise ValueError('invalid correspondence cells')
    def directional(x):
        labels = []
        for row in x:
            positive = np.flatnonzero(row == 1)
            if len(positive) > 1:
                labels.append(-2)
            elif len(row) and (row >= 0).all():
                labels.append(int(positive[0]) if len(positive) else -1)
            else:
                labels.append(-2)
        return labels
    return directional(matrix), directional(matrix.T)


def pair_cells(pair, associations_a, associations_b, gt_a, gt_b, metric, clearance=.45):
    cells = np.full((len(associations_a), len(associations_b)), -1, dtype=np.int64)
    positives = {(x['source_interface_a'], x['source_interface_b']) for x in pair['portal_correspondences']}
    for i, a in enumerate(associations_a):
        for j, b in enumerate(associations_b):
            if a['status'] != 'associated' or b['status'] != 'associated':
                continue
            ai, bi = a['interface_id'], b['interface_id']
            if (ai, bi) in positives:
                cells[i, j] = 1; continue
            # Absence of a link is not negative evidence. Only an explicit
            # certificate or a conservative lower bound on physical separation.
            if pair['physical_portal_label'] == 'no_shared' and pair.get('evidence'):
                cells[i, j] = 0; continue
            x, y = gt_a[ai].get('floor_endpoints'), gt_b[bi].get('floor_endpoints')
            if metric is None or x is None or y is None:
                continue
            x, y = np.asarray(x), np.asarray(y)
            gap = np.linalg.norm(x.mean(0)-y.mean(0)) - .5*(np.linalg.norm(x[1]-x[0])+np.linalg.norm(y[1]-y[0]))
            if np.isfinite(gap) and gap*metric > clearance:
                cells[i, j] = 0
    return cells


def build_weak_supervision(cache, dataset_root, output, review_file=None, *, protocol=None, audit_only=False):
    root, output = Path(dataset_root).resolve(), Path(output).resolve()
    if output.exists() or root in output.parents or cache.root in output.parents:
        raise ValueError('use a new independent supervision output file')
    if audit_only and (protocol is None or 'audit' not in protocol):
        raise ValueError('audit export requires a four-role protocol')
    if protocol is not None:
        from portallayout.datasets.floor_auxiliary_supervision import validate_cache_house_roles
        allowed = {'audit'} if audit_only else {'fit', 'selection', 'calibration'}
        validate_cache_house_roles(cache, root, protocol, allowed)
    samples = {s['sample_id']: s for s in read_jsonl(root/'samples/all_panoramas.jsonl')}
    reviews = json.loads(Path(review_file).read_text()) if review_file else {'pairs': {}}
    if review_file and reviews.get('source_snapshot') != cache.snapshot:
        raise ValueError('review snapshot mismatch')
    sources, rows, associations = {}, [], {}
    cfg = {'iou_threshold': .5, 'margin': .1, 'negative_clearance_m': .45}
    for floor in cache.floors:
        sample = samples.get(floor['sample_id'])
        if sample is None:
            raise ValueError('cached floor is not an all-panoramas dataset sample')
        base = safe_path(root, 'labels/'+floor['floor_uid'])
        for name in ('observations_gt.jsonl', 'portal_links.jsonl', 'floor_gt.json'):
            sources[str(base/name)] = content_hash(base/name)
        annotation = {x['observation_id']: x for x in read_jsonl(base/'observations_gt.jsonl')}
        floor_gt = json.loads((base/'floor_gt.json').read_text())
        gt = {}
        for uid in floor['observation_order']:
            if uid in cache.records:
                associations[uid], gt[uid] = associate_candidates(cache.records[uid], annotation[uid])
        for pair in iter_floor_pairs(base, floor['observation_order']):
            aid, bid = pair['observation_a'], pair['observation_b']
            if aid not in gt or bid not in gt:
                continue
            cells = pair_cells(pair, associations[aid], associations[bid], gt[aid], gt[bid],
                               floor_gt['metric_scale'], cfg['negative_clearance_m'])
            original = cells.copy()
            ids_a = [x['candidate_id'] for x in associations[aid]]
            ids_b = [x['candidate_id'] for x in associations[bid]]
            review = reviews['pairs'].get(pair['pair_id'])
            reviewed = np.full_like(cells, -1)
            if review is not None:
                if (review.get('candidate_ids_a') != ids_a or review.get('candidate_ids_b') != ids_b or
                        not review.get('reviewer') or not review.get('reviewed_at')):
                    raise ValueError('review requires exact candidate IDs, reviewer and date')
                reviewed = np.asarray(review['cells'], dtype=np.int64).reshape(cells.shape)
                if not np.isin(reviewed, [-1, 0, 1]).all() or (reviewed == 1).sum(0).max(initial=0) > 1 or (reviewed == 1).sum(1).max(initial=0) > 1:
                    raise ValueError('review violates one-to-one correspondence')
                cells[reviewed >= 0] = reviewed[reviewed >= 0]
            ta, tb = complete_targets(cells)
            ra, rb = complete_targets(reviewed)
            rows.append({'pair_id': pair['pair_id'], 'floor_uid': floor['floor_uid'],
                'observation_a': aid, 'observation_b': bid, 'house_id': sample['house_id'],
                'official_split': sample['official_split'], 'training_subset': sample['training_subset'],
                'candidate_ids_a': ids_a, 'candidate_ids_b': ids_b, 'cells': cells.tolist(),
                'weak_cells': original.tolist(), 'reviewed_cells': reviewed.tolist(),
                'target_a': ta, 'target_b': tb, 'review_target_a': ra, 'review_target_b': rb,
                'label_source': 'geometry_weak' if review is None else 'geometry_weak_with_review_overlay',
                'review': review, 'visual_labels_in_original_dataset': 'unknown',
                'source_cache_keys': [cache.records[aid]['cache_key'], cache.records[bid]['cache_key']]})
    stats = {'pairs': len(rows), 'supervised_pairs': sum(any(v != -2 for v in r['target_a']+r['target_b']) for r in rows),
             'positive_cells': sum(sum(v == 1 for v in row) for r in rows for row in r['cells']),
             'negative_cells': sum(sum(v == 0 for v in row) for r in rows for row in r['cells']),
             'unknown_cells': sum(sum(v == -1 for v in row) for r in rows for row in r['cells']),
             'association_status': {s: sum(a['status'] == s for rows_ in associations.values() for a in rows_)
                                    for s in ('associated', 'ambiguous', 'missing_annotation')},
             'reviewed_pairs': sum(r['review'] is not None for r in rows)}
    record = {'schema_version': 'opening-pair-supervision.v1', 'label_source': 'geometry_weak',
              'source_snapshot': cache.snapshot, 'source_contract': cache.fingerprint,
              'source_files_sha256': sources, 'review_sha256': content_hash(review_file) if review_file else None,
              'config': cfg, 'associations': associations, 'pairs': rows, 'statistics': stats}
    if protocol is not None:
        record['house_protocol'] = protocol
        record['usage'] = 'audit_only' if audit_only else 'training_calibration'
    record['fingerprint'] = fingerprint(record)
    if any(content_hash(p) != h for p, h in sources.items()):
        raise ValueError('annotation source changed during construction')
    cache.check_unchanged()
    write_json_atomic(output, record)
    return record
