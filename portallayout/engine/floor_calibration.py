"""Offline geometry-development calibration; unknown evidence remains masked."""
import math
from pathlib import Path
import numpy as np
from portallayout.models.floor_auxiliary import FEATURE_VERSION
from portallayout.engine.floor_training import AuxiliaryFeatureDataset, predict_auxiliary
from portallayout.engine.floor_layout_pipeline import FloorModelBundle
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def wilson(successes, total, z=1.959963984540054):
    if not total: return None
    p = successes/total; denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def thresholds(probabilities):
    if not probabilities: return [1.]
    values = np.asarray(probabilities, float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('calibration probabilities must be finite and in [0, 1]')
    # Every distinct score is a decision boundary. A quantile grid can miss a
    # narrow feasible interval and incorrectly declare calibration infeasible.
    return sorted(set([0., 1.]+values.tolist()))


def binary_workpoint(rows, min_precision=.95, max_fpr=.01, minimum_houses=5):
    known = [r for r in rows if r['target'] >= 0]
    positives = sum(r['target'] == 1 for r in known); negatives = sum(r['target'] == 0 for r in known)
    houses = sorted({r['house_id'] for r in known})
    base = {'positives': positives, 'negatives': negatives, 'houses': houses,
            'unknown': len(rows)-len(known), 'min_precision': min_precision, 'max_fpr': max_fpr}
    if not positives or not negatives or len(houses) < minimum_houses:
        return dict(base, status='calibration_infeasible', reason='insufficient_class_or_house_support')
    feasible = []
    for threshold in thresholds([r['probability'] for r in known]):
        selected = [r for r in known if r['probability'] >= threshold]
        tp = sum(r['target'] == 1 for r in selected); fp = len(selected)-tp
        if not selected: continue
        precision, recall, fpr = tp/len(selected), tp/positives, fp/negatives
        if precision >= min_precision and (max_fpr is None or fpr <= max_fpr):
            feasible.append((recall, precision, threshold, tp, fp, fpr))
    if not feasible: return dict(base, status='calibration_infeasible', reason='no_threshold_meets_operating_point')
    recall, precision, threshold, tp, fp, fpr = max(feasible)
    return dict(base, status='calibrated', threshold=threshold, precision=precision, recall=recall, false_positive_rate=fpr,
                accepted_known=tp+fp, accepted_unknown=sum(r['target'] < 0 and r['probability'] >= threshold for r in rows),
                precision_ci95=wilson(tp, tp+fp), false_positive_rate_ci95=wilson(fp, negatives))


def opening_workpoint(pairs, minimum_houses=5):
    pairs = [p for p in pairs if p['role'] == 'calibration']
    known = [dict(m, house_id=p['house_id']) for p in pairs for m in p['matches'] if m['target'] >= 0]
    negative_pairs = [p for p in pairs if p['known_negative_pair']]
    positive_count = sum(m['target'] == 1 for m in known)
    houses = sorted({p['house_id'] for p in pairs})
    base = {'houses': houses, 'positive_retrieved_matches': positive_count, 'known_negative_pairs': len(negative_pairs),
            'precision_semantics': 'geometry_weak_correspondence_on_known_cells', 'minimum_houses': minimum_houses}
    if len(houses) < minimum_houses or not positive_count or not negative_pairs:
        return dict(base, status='calibration_infeasible', reason='insufficient_class_or_house_support')
    feasible = []
    for threshold in thresholds([m['confidence'] for m in known]):
        selected = [m for m in known if m['confidence'] >= threshold]
        if not selected: continue
        tp = sum(m['target'] == 1 for m in selected); fp = len(selected)-tp
        bad_pairs = sum(any(m['confidence'] >= threshold for m in p['matches']) for p in negative_pairs)
        if tp/len(selected) >= .95 and bad_pairs/len(negative_pairs) <= .01:
            feasible.append((tp/positive_count, tp/len(selected), threshold, tp, fp, bad_pairs))
    if not feasible: return dict(base, status='calibration_infeasible', reason='no_threshold_meets_operating_point')
    recall, precision, threshold, tp, fp, bad_pairs = max(feasible)
    all_accepted = [m for p in pairs for m in p['matches'] if m['confidence'] >= threshold]
    return dict(base, status='calibrated', threshold=threshold, precision=precision, recall_of_retrieved_positive_matches=recall,
        negative_pair_false_acceptance=bad_pairs/len(negative_pairs), accepted_known=tp+fp,
        accepted_unknown=sum(m['target'] < 0 for m in all_accepted), precision_ci95=wilson(tp, tp+fp),
        false_acceptance_ci95=wilson(bad_pairs, len(negative_pairs)))


def strict_opening_workpoint(pairs):
    """Round-two progression policy; support counts accepted cross-room TPs."""
    from portallayout.evaluation.opening_matching import summary
    rows = [r for r in pairs if r['role'] == 'calibration']
    cross = [r for r in rows if r['relationship'] == 'cross_room']
    base = dict(policy='opening-cross-room.v2', minimum_cross_recall=.1, minimum_cross_true=20,
                minimum_cross_positive_houses=5, minimum_precision=.95, maximum_negative_pair_far=.01,
                all_positive_cells=sum(r['positive_cells'] for r in rows),
                cross_positive_cells=sum(r['positive_cells'] for r in cross),
                known_negative_pairs=sum(r['known_negative_pair'] for r in rows))
    if not base['cross_positive_cells'] or not base['known_negative_pairs']:
        return dict(base, status='calibration_infeasible', reason='insufficient_class_support')
    candidates = []; failure_counts = {}
    for threshold in thresholds([m['confidence'] for r in rows for m in r['matches'] if m['target'] >= 0]):
        overall, subgroup = summary(rows, threshold), summary(cross, threshold)
        checks = dict(overall_precision=(overall['precision_known'] or 0.) >= .95,
                      cross_precision=(subgroup['precision_known'] or 0.) >= .95,
                      negative_pair_far=overall['negative_pair_false_acceptance'] <= .01,
                      cross_recall=(subgroup['recall_all_weak_positives'] or 0.) >= .1,
                      cross_true_support=subgroup['accepted_true'] >= 20,
                      cross_house_support=subgroup['accepted_positive_houses'] >= 5)
        for name, passed in checks.items(): failure_counts[name] = failure_counts.get(name, 0)+int(not passed)
        if all(checks.values()):
            candidates.append((subgroup['recall_all_weak_positives'], overall['precision_known'], threshold,
                               overall, subgroup))
    if not candidates:
        return dict(base, status='calibration_infeasible', reason='no_threshold_meets_quality_and_support',
                    failed_threshold_counts=failure_counts)
    _, _, threshold, overall, cross_stats = max(candidates, key=lambda x: x[:3])
    return dict(base, status='calibrated', threshold=threshold, overall=overall, cross_room=cross_stats,
                precision_ci95=wilson(overall['accepted_true'], overall['accepted_true']+overall['accepted_false']),
                cross_precision_ci95=wilson(cross_stats['accepted_true'], cross_stats['accepted_true']+cross_stats['accepted_false']),
                cross_recall_ci95=wilson(cross_stats['accepted_true'], cross_stats['positive_cells']),
                false_acceptance_ci95=wilson(overall['false_accepted_negative_pairs'], overall['known_negative_pairs']))


def pose_workpoint(rows, minimum_houses=5):
    groups = {}
    for row in rows: groups.setdefault(row['pair_id'], []).append(row)
    ranked = {}
    for key, values in groups.items():
        unique = []
        for row in sorted(values, key=lambda r: (-r['probability'], r['hypothesis_id'])):
            if not any(np.allclose(row['T_B_to_A'], r['T_B_to_A'], atol=1e-5, rtol=1e-5) for r in unique): unique.append(row)
        ranked[key] = unique
    positives = sum(any(r['target'] == 1 for r in g) for g in ranked.values())
    negative_groups = [g for g in ranked.values() if all(r['target'] == 0 for r in g)]
    houses = sorted({r['house_id'] for r in rows if r['target'] >= 0})
    base = {'houses': houses, 'positive_pairs_with_recalled_hypothesis': positives,
            'known_negative_pairs': len(negative_groups), 'minimum_houses': minimum_houses}
    if not positives or not negative_groups or len(houses) < minimum_houses:
        return dict(base, status='calibration_infeasible', reason='insufficient_class_or_house_support')
    feasible = []
    for margin in (0., .05, .1, .2):
        for threshold in thresholds([g[0]['probability'] for g in ranked.values()]):
            accepted = []
            for group in ranked.values():
                best, runner = group[0], group[1]['probability'] if len(group) > 1 else 0.
                if best['probability'] >= threshold and best['probability']-runner >= margin and best['independent_evidence']:
                    accepted.append(best)
            selected = [r for r in accepted if r['target'] >= 0]
            if not selected: continue
            tp = sum(r['target'] == 1 for r in selected); fp = len(selected)-tp
            accepted_ids = {r['pair_id'] for r in accepted}
            bad = sum(g[0]['pair_id'] in accepted_ids for g in negative_groups)
            if tp/len(selected) >= .95 and bad/len(negative_groups) <= .01:
                feasible.append((tp/positives, tp/len(selected), threshold, margin, tp, fp, bad, len(accepted)-len(selected)))
    if not feasible: return dict(base, status='calibration_infeasible', reason='no_threshold_margin_meets_operating_point')
    recall, precision, threshold, margin, tp, fp, bad, unknown = max(feasible)
    return dict(base, status='calibrated', threshold=threshold, margin=margin, precision=precision,
        recall_of_recalled_positive_pairs=recall, negative_pair_false_acceptance=bad/len(negative_groups),
        accepted_known=tp+fp, accepted_unknown=unknown, precision_ci95=wilson(tp, tp+fp),
        false_acceptance_ci95=wilson(bad, len(negative_groups)))


def calibrate_floor_models(supervision, pair_checkpoint, auxiliary_checkpoint, output, device='cpu', minimum_houses=5,
                           opening_report=None):
    output = Path(output)
    if output.exists(): raise ValueError('calibration output must be new')
    dataset = AuxiliaryFeatureDataset(supervision)
    bundle = FloorModelBundle({'pair_checkpoint': str(pair_checkpoint), 'auxiliary_checkpoint': str(auxiliary_checkpoint)}, device)
    if bundle.pair_state['training_status'] != 'trained' or not any(bundle.auxiliary_trained(t) for t in ('pose','room')):
        raise ValueError('calibration requires a trained matcher and at least one full-trained auxiliary head')
    if dataset.data['pair_checkpoint_sha256'] != bundle.auxiliary_state['pair_checkpoint_sha256']:
        raise ValueError('calibration features and model pair checkpoint differ')
    protocol = dataset.data['house_protocol']
    if dataset.data.get('room_feature_source') == 'predicted_pgo_components':
        from portallayout.engine.floor_training import _state_hash
        if dataset.data['room_pose_model_hash'] != _state_hash(bundle.pose_model):
            raise ValueError('room calibration pose verifier changed since PGO export')
    if protocol != bundle.auxiliary_state['training_contract']['house_protocol']:
        raise ValueError('calibration house protocol mismatch')
    training_houses = set(bundle.pair_state['training_contract'].get('fit_houses', []))
    training_houses |= set(bundle.auxiliary_state['training_contract']['house_protocol']['fit'])
    if training_houses & set(protocol['calibration']): raise ValueError('calibration/train house leakage')
    if opening_report:
        import json
        report = json.loads(Path(opening_report).read_text()); claimed = report.pop('fingerprint')
        if fingerprint(report) != claimed: raise ValueError('opening report fingerprint mismatch')
        if (report['pair_checkpoint_sha256'] != content_hash(pair_checkpoint) or
                report['source_snapshot'] != dataset.data['source_snapshot'] or report['house_protocol'] != protocol):
            raise ValueError('opening report model/cache/protocol mismatch')
        opening = report['calibration']
        if opening.get('policy') != 'opening-cross-room.v2': raise ValueError('strict opening policy required')
    else:
        if bundle.pair_state['training_contract'].get('selection_metric') == 'cross_room_ap':
            raise ValueError('task-selected models require a bound strict opening report')
        opening = opening_workpoint(dataset.data['opening_pairs'], minimum_houses)
    result = {'schema_version': 'floor-calibration.v1', 'track': 'geometry_dev', 'weights_hash': bundle.weights_hash,
              'feature_version': FEATURE_VERSION, 'house_protocol': protocol,
              'supervision_fingerprint': dataset.data['fingerprint'], 'opening': opening,
              'opening_report_sha256': content_hash(opening_report) if opening_report else None}
    result['room_feature_source'] = dataset.data.get('room_feature_source', 'pair_hypotheses_only')
    for task, model in (('pose', bundle.pose_model), ('room', bundle.room_model)):
        if not bundle.auxiliary_trained(task):
            result[task] = {'status': 'calibration_infeasible', 'reason': 'auxiliary_head_not_trained'}
            continue
        rows = [r for r in dataset.data['records'] if r['task'] == task and r['role'] == 'calibration']
        probabilities = predict_auxiliary(model, dataset, rows, device)
        values = [dict(r, probability=p) for r, p in zip(rows, probabilities)]
        if task == 'pose':
            if result['opening']['status'] != 'calibrated':
                result[task] = {'status': 'calibration_infeasible', 'reason': 'opening_gate_infeasible'}
                continue
            if any('opening_score' not in r for r in values):
                raise ValueError('pose calibration requires seed opening scores from current feature export')
            threshold = result['opening']['threshold']
            gated = [r for r in values if r['opening_score'] >= threshold]
            result[task] = pose_workpoint(gated, minimum_houses)
            result[task]['opening_threshold'] = threshold
            result[task]['hypotheses_before_opening_gate'] = len(values)
            result[task]['hypotheses_after_opening_gate'] = len(gated)
        else:
            result[task] = binary_workpoint(values, .98, None, minimum_houses)
    result['status'] = 'calibrated' if all(result[t]['status'] == 'calibrated' for t in ('opening', 'pose', 'room')) else 'calibration_infeasible'
    result['confidence_interval_note'] = 'Wilson intervals are descriptive; house-correlated final results require house bootstrap.'
    write_json_atomic(output, result)
    return result
