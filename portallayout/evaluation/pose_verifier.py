"""Offline, tied-score AP and pair-level exploratory pose working points."""
from collections import defaultdict
import numpy as np
from portallayout.engine.pose_verifier_policy import accepted_best


def ap(rows):
    known = [r for r in rows if r['target'] >= 0]
    positives = sum(r['target'] == 1 for r in known)
    if not positives: return None
    groups = defaultdict(lambda: [0, 0])
    for r in known: groups[float(r['probability'])][1-r['target']] += 1
    tp = fp = 0; area = 0.
    for score, (yes, no) in sorted(groups.items(), reverse=True):
        tp += yes; fp += no; area += yes/positives*tp/(tp+fp)
    return area


def quality(rows):
    known = [r for r in rows if r['target'] >= 0]
    per_house = {h: ap([r for r in rows if r['house_id'] == h and r.get('relationship') == 'cross_room'])
                 for h in sorted({r['house_id'] for r in rows})}
    values = [v for v in per_house.values() if v is not None]
    losses = {}
    for label in (0, 1):
        p = np.clip([r['probability'] for r in known if r['target'] == label], 1e-7, 1-1e-7)
        if len(p): losses[str(label)] = float(np.mean(-np.log(p if label else 1-p)))
    groups = defaultdict(list)
    for row in rows: groups[row['pair_id']].append(row)
    positive_groups = [g for g in groups.values() if any(r['target'] == 1 for r in g)]
    selected = [sorted(g, key=lambda r: (-r['probability'], r['hypothesis_id']))[0] for g in positive_groups]
    return dict(cross_room_house_macro_ap=float(np.mean(values)) if values else None,
                cross_room_ap_by_house=per_house, overall_ap=ap(rows),
                same_room_ap=ap([r for r in rows if r.get('relationship') == 'same_room']),
                balanced_bce=float(np.mean(list(losses.values()))) if losses else None,
                class_bce=losses, positive_pairs=len(positive_groups),
                correct_selection_rate=sum(r['target'] == 1 for r in selected)/len(selected) if selected else None,
                hypotheses=len(rows), ignored=sum(r['target'] < 0 for r in rows))


def groups_at_gate(rows, opening_threshold=.7):
    groups = defaultdict(list)
    for r in rows:
        if r['opening_score'] >= opening_threshold:
            groups[r['pair_id']].append(dict(r, verifier_probability=r['probability'],
                nonfit_evidence_available=r['independent_evidence']))
    return {p: sorted(g, key=lambda r: (-r['probability'], r['hypothesis_id'])) for p, g in groups.items()}


def assess_workpoint(rows, threshold, margin, opening_threshold=.7):
    groups = groups_at_gate(rows, opening_threshold)
    accepted = [accepted_best(g, threshold, margin) for g in groups.values()]
    accepted = [r for r in accepted if r is not None]
    def summary(scope=None):
        gg = [g for g in groups.values() if scope is None or g[0].get('relationship') == scope]
        aa = [r for r in accepted if scope is None or r.get('relationship') == scope]
        positive = sum(any(r['target'] == 1 for r in g) for g in gg)
        negative = [g for g in gg if all(r['target'] == 0 for r in g)]
        ids = {r['pair_id'] for r in aa}
        tp = sum(r['target'] == 1 for r in aa); fp = sum(r['target'] == 0 for r in aa)
        bad = sum(g[0]['pair_id'] in ids for g in negative)
        return dict(positive_pairs=positive, correct_edges=tp, incorrect_edges=fp,
            accepted_unknown=sum(r['target'] < 0 for r in aa),
            correct_houses=len({r['house_id'] for r in aa if r['target'] == 1}),
            precision=tp/(tp+fp) if tp+fp else None, recall=tp/positive if positive else None,
            negative_pairs=len(negative), false_accepted_negative_pairs=bad,
            negative_pair_far=bad/len(negative) if negative else None)
    overall, cross = summary(), summary('cross_room')
    checks = dict(overall_precision=(overall['precision'] or 0.) >= .95,
                  cross_precision=(cross['precision'] or 0.) >= .95,
                  negative_pair_far=overall['negative_pair_far'] is not None and overall['negative_pair_far'] <= .01,
                  correct_edges=overall['correct_edges'] >= 20, correct_houses=overall['correct_houses'] >= 5)
    return dict(threshold=threshold, margin=margin, opening_threshold=opening_threshold, overall=overall,
                cross_room=cross, same_room=summary('same_room'), checks=checks, passed=all(checks.values()))


def select_workpoint(rows):
    grid = [assess_workpoint(rows, t, m) for t in (.5,.6,.7,.8,.9,.95,.99) for m in (0.,.05,.1,.2)]
    feasible = [x for x in grid if x['passed']]
    report = dict(status='selection_workpoint_infeasible', opening_threshold=.7,
                  calibrated=False, grid=grid, reason='no_workpoint_meets_all_constraints')
    if feasible:
        best = max(feasible, key=lambda x: (x['cross_room']['recall'], x['overall']['recall'],
                    x['overall']['precision'], x['threshold'], x['margin']))
        report.update(best, status='selection_workpoint_found', reason=None)
    return report
