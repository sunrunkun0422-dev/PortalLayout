"""Offline opening metrics and lightweight all-pair evaluation, without pose BEV."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from portallayout.models.opening_pair_matcher import mutual_matches
from portallayout.engine.opening_pair_training import pair_objective
from portallayout.utils.room_feature_cache import content_hash


def room_scopes(rows, dataset_root):
    """Membership is consumed only after prediction for offline stratification."""
    scopes, hashes = {}, {}
    grouped = defaultdict(list)
    for row in rows: grouped[row['floor_uid']].append(row)
    for uid, pairs in grouped.items():
        from portallayout.datasets.zind_floor_dataset import safe_path
        path = safe_path(Path(dataset_root), 'labels/'+uid+'/observations_gt.jsonl')
        annotations = {r['observation_id']: r for r in map(json.loads, path.read_text().splitlines())}
        hashes[str(path)] = content_hash(path)
        for row in pairs:
            a, b = [annotations[row['observation_'+s]] for s in ('a', 'b')]
            scopes[row['pair_id']] = ('same_room' if a['complete_room_uid'] == b['complete_room_uid'] else 'cross_room') if (
                a.get('membership_status') == b.get('membership_status') == 'confirmed') else 'unknown_room'
    return scopes, hashes


def average_precision(rows):
    positives = sum(r['positive_cells'] for r in rows)
    if not positives: return None
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        for match in row['matches']:
            if match['target'] >= 0: groups[match['confidence']][1-match['target']] += 1
    tp = fp = 0; area = 0.
    for _, (yes, no) in sorted(groups.items(), reverse=True):
        tp += yes; fp += no
        area += yes/positives * tp/(tp+fp)
    return area


def summary(rows, threshold=0.):
    selected = [(r, m) for r in rows for m in r['matches'] if m['confidence'] >= threshold]
    tp = sum(m['target'] == 1 for _, m in selected); fp = sum(m['target'] == 0 for _, m in selected)
    positives = sum(r['positive_cells'] for r in rows)
    negatives = [r for r in rows if r['known_negative_pair']]
    bad = sum(any(m['confidence'] >= threshold for m in r['matches']) for r in negatives)
    diagnostics = [d for r in rows for d in r.get('positive_diagnostics', [])]
    ranks = {key: sum(d.get(key, False) for d in diagnostics) for key in
             ('raw_mutual_top1', 'raw_bidirectional_top3', 'assignment_mutual_without_dustbin',
              'mutual_with_dustbin', 'dustbin_wins_either_direction', 'assignment_supervised')}
    supervision = {}
    for available in (True, False):
        values = [d for d in diagnostics if d['assignment_supervised'] == available]
        supervision['complete_positive_row' if available else 'partial_positive_only'] = dict(
            positive_cells=len(values), retrieved=sum(d.get('mutual_with_dustbin', False) for d in values))
    losses = {}
    for group in sorted({r.get('supervision_group', 'unknown') for r in rows}):
        selected_rows = [r for r in rows if r.get('supervision_group', 'unknown') == group]
        losses[group] = {'pairs': len(selected_rows)}
        for name in ('assignment_loss', 'auxiliary_loss', 'objective_loss'):
            values = [r[name] for r in selected_rows if r.get(name) is not None]
            losses[group][name] = float(np.mean(values)) if values else None
    return dict(pairs=len(rows), houses=len({r['house_id'] for r in rows}), threshold=threshold,
                positive_cells=positives, accepted_true=tp, accepted_false=fp,
                accepted_unknown=sum(m['target'] < 0 for _, m in selected),
                accepted_positive_houses=len({r['house_id'] for r, m in selected if m['target'] == 1}),
                precision_known=tp/(tp+fp) if tp+fp else None,
                recall_all_weak_positives=tp/positives if positives else None,
                f1_known=2*tp/(positives+tp+fp) if positives+tp+fp else None,
                known_negative_pairs=len(negatives), false_accepted_negative_pairs=bad,
                negative_pair_false_acceptance=bad/len(negatives) if negatives else None,
                numerical_status=dict(Counter(r['numerical_status'] for r in rows)), positive_stages=ranks,
                positive_supervision_groups=supervision, losses_by_supervision=losses,
                unretrieved_positive_cells=positives-sum(m['target'] == 1 for r in rows for m in r['matches']))


def quality_metrics(rows):
    cross = [r for r in rows if r['relationship'] == 'cross_room']
    houses = {h: [r for r in cross if r['house_id'] == h] for h in sorted({r['house_id'] for r in rows})}
    per_house = {h: average_precision(r) for h, r in houses.items()}
    valid = [v for v in per_house.values() if v is not None]
    loss_groups = defaultdict(list)
    for row in rows:
        value = row.get('objective_loss')
        if value is not None: loss_groups['positive' if row['positive_cells'] else 'no_known_positive'].append(value)
    group_means = {k: float(np.mean(v)) for k, v in loss_groups.items()}
    assignment = [r['assignment_loss'] for r in rows if r.get('assignment_loss') is not None]
    return dict(cross_room_house_macro_ap=float(np.mean(valid)) if valid else None,
                cross_room_ap_by_house=per_house, overall_ap=average_precision(rows),
                balanced_loss=float(np.mean(list(group_means.values()))) if group_means else None,
                losses_by_pair_class=group_means,
                assignment_loss=float(np.mean(assignment)) if assignment else None,
                pair_count=len(rows), zero_positive_house_false_acceptance={h: summary(v)
                    for h, v in houses.items() if per_house[h] is None})


def selection_key(metrics):
    def finite(value, fallback): return fallback if value is None else value
    return (finite(metrics['cross_room_house_macro_ap'], -1.), finite(metrics['overall_ap'], -1.),
            -finite(metrics['balanced_loss'], float('inf')))


def score_record(output, row, relationship, objective='assignment', auxiliary_weight=.5, margin=.2):
    cells = np.asarray(row['weak_cells'], dtype=int).reshape(len(row['candidate_ids_a']), len(row['candidate_ids_b']))
    if not np.isin(cells, [-1, 0, 1]).all(): raise ValueError('invalid weak labels')
    result = {k: row[k] for k in ('pair_id', 'floor_uid', 'house_id', 'observation_a', 'observation_b',
                                  'candidate_ids_a', 'candidate_ids_b')}
    result.update(relationship=relationship, positive_cells=int((cells == 1).sum()),
                  known_negative_pair=bool(cells.size and (cells == 0).all()), matches=[],
                  numerical_status=output['status'], objective_loss=None, assignment_loss=None,
                  auxiliary_loss=None, positive_diagnostics=[])
    targets = row['target_a']+row['target_b']
    result['supervision_group'] = ('complete_positive_row' if any(t >= 0 for t in targets) else
                                 'partial_positive_only' if (cells == 1).any() else
                                 'unmatched_row' if -1 in targets else 'unknown_assignment')
    if output['status'] != 'assigned': return result
    loss, parts = pair_objective(output, row, objective, auxiliary_weight, margin)
    result.update(objective_loss=float(loss) if loss is not None else None,
                  assignment_loss=float(parts['assignment']) if parts['assignment'] is not None else None,
                  auxiliary_loss=float(parts['auxiliary']) if parts['auxiliary'] is not None else None)
    a, b = output['indices_a'].cpu().tolist(), output['indices_b'].cpu().tolist()
    if ([output['candidate_ids_a'][i] for i in a] != [row['candidate_ids_a'][i] for i in a] or
            [output['candidate_ids_b'][j] for j in b] != [row['candidate_ids_b'][j] for j in b]):
        raise ValueError('prediction/label candidate identity mismatch')
    score = output['scores'].detach().cpu().numpy()
    ab = output['conditional_ab'].detach().cpu().numpy(); ba = output['conditional_ba'].detach().cpu().numpy()
    result.update(scores=score.tolist(), conditional_ab=ab.tolist(), conditional_ba=ba.tolist(),
                  indices_a=a, indices_b=b, dustbin_score=float(output['dustbin_score']),
                  iterations=output['iterations'], marginal_error=output['marginal_error'])
    for m in mutual_matches(output, 0.):
        i, j = m['index_a'], m['index_b']
        result['matches'].append(dict(m, target=int(cells[i, j]),
            candidate_id_a=row['candidate_ids_a'][i], candidate_id_b=row['candidate_ids_b'][j]))
    amap, bmap = {i: x for x, i in enumerate(a)}, {j: y for y, j in enumerate(b)}
    matched = {(m['index_a'], m['index_b']) for m in result['matches']}
    for i, j in zip(*np.where(cells == 1)):
        d = dict(index_a=int(i), index_b=int(j), assignment_supervised=bool(
                 row['target_a'][i] == j or row['target_b'][j] == i))
        if i in amap and j in bmap:
            x, y = amap[i], bmap[j]
            d.update(raw_mutual_top1=bool(score[x].argmax() == y and score[:, y].argmax() == x),
                     raw_bidirectional_top3=bool((score[x] > score[x, y]).sum() < 3 and (score[:, y] > score[x, y]).sum() < 3),
                     assignment_mutual_without_dustbin=bool(ab[x, :-1].argmax() == y and ba[y, :-1].argmax() == x),
                     mutual_with_dustbin=(i, j) in matched,
                     dustbin_wins_either_direction=bool(ab[x].argmax() == len(b) or ba[y].argmax() == len(a)))
        result['positive_diagnostics'].append(d)
    return result


def evaluate_matching(cache, encoder, matcher, rows, scopes, device, objective='assignment',
                      auxiliary_weight=.5, margin=.2, output_file=None):
    from portallayout.engine.floor_layout_pipeline import encode_floor
    encoder.eval(); matcher.eval()
    grouped = defaultdict(list)
    for row in rows: grouped[row['floor_uid']].append(row)
    records = []; start = time.perf_counter()
    stream = None
    if output_file:
        path = Path(output_file); path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('x')
    try:
        for floor, targets in sorted(grouped.items()):
            ids = sorted({r['observation_'+s] for r in targets for s in ('a', 'b')})
            _, views, _ = encode_floor(cache, ids, SimpleNamespace(encoder=encoder, device=device))
            with torch.no_grad():
                for row in sorted(targets, key=lambda r: r['pair_id']):
                    a, b = row['observation_a'], row['observation_b']
                    if a not in views or b not in views:
                        output = {'status': 'missing_prediction'}
                    else:
                        output = matcher(views[a], views[b])
                    record = score_record(output, row, scopes[row['pair_id']], objective, auxiliary_weight, margin)
                    if stream: stream.write(json.dumps(record, allow_nan=False)+'\n')
                    # Retain small metrics only; raw matrices are streamed once for audit.
                    records.append({k: v for k, v in record.items() if k not in
                                    ('scores', 'conditional_ab', 'conditional_ba')})
            del views
        return records, dict(quality_metrics(records), elapsed_seconds=time.perf_counter()-start)
    finally:
        if stream: stream.close()
