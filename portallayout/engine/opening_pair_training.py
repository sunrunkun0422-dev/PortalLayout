"""Train new modules from frozen visual features, with separately loaded labels."""
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from portallayout.models.opening_representation import OpeningRepresentationEncoder
from portallayout.models.opening_pair_matcher import OpeningPairMatcher, assignment_loss, mutual_matches
from portallayout.utils.opening_representation_adapter import prepare_opening_representation_batch
from portallayout.utils.frozen_opening_cache import pair_views
from portallayout.utils.opening_pair_checkpoint import (checkpoint_payload, load_checkpoint, save_checkpoint, weights_hash)
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def read_supervision(path, cache, *, allow_audit=False):
    data = json.loads(Path(path).read_text())
    claimed = data.pop('fingerprint')
    if fingerprint(data) != claimed or data['source_snapshot'] != cache.snapshot:
        raise ValueError('supervision fingerprint/source mismatch')
    data['fingerprint'] = claimed
    if data.get('usage') == 'audit_only' and not allow_audit:
        raise ValueError('audit-only supervision cannot be used for training')
    if 'house_protocol' in data:
        from portallayout.datasets.floor_auxiliary_supervision import validate_supervision_house_roles
        roles = {'audit'} if data.get('usage') == 'audit_only' else {'fit', 'selection', 'calibration'}
        validate_supervision_house_roles(data, data['house_protocol'], roles)
    for row in data['pairs']:
        for side in ('a', 'b'):
            uid = row['observation_'+side]
            ids = [x['candidate_id'] for x in cache.records[uid]['prediction']['openings']]
            if ids != row['candidate_ids_'+side]:
                raise ValueError('supervision candidate identity mismatch')
    return data


def encode_rows(cache, encoder, rows, device):
    ids = sorted({row['observation_'+side] for row in rows for side in ('a', 'b')})
    loaded = [cache.load(uid) for uid in ids]
    prepared = prepare_opening_representation_batch([r for r, _ in loaded], [a for _, a in loaded], device)
    encoded = encoder(*prepared.encoder_inputs(), return_diagnostics=True)
    return dict(zip(ids, pair_views(encoded, prepared)))


PAIR_OBJECTIVES = ('assignment', 'positive_margin', 'known_cells')


def pair_objective(output, row, objective='assignment', auxiliary_weight=.5, margin=.2):
    """Additional *cell* evidence never completes unknown assignment rows."""
    if objective not in PAIR_OBJECTIVES or auxiliary_weight < 0 or margin < 0:
        raise ValueError('invalid pair objective')
    assignment = assignment_loss(output, row['target_a'], row['target_b'])
    auxiliary = None
    if objective != 'assignment' and output['status'] == 'assigned':
        scores, dustbin = output['scores'], output['dustbin_score']
        cells = torch.as_tensor(row['weak_cells'], device=scores.device, dtype=torch.long)
        cells = cells.reshape(len(row['candidate_ids_a']), len(row['candidate_ids_b']))
        if not torch.all((cells >= -1) & (cells <= 1)):
            raise ValueError('invalid weak cell labels')
        cells = cells[output['indices_a']][:, output['indices_b']]
        positive, negative = cells == 1, cells == 0
        if objective == 'positive_margin' and positive.any():
            from portallayout.models.cross_scene_matcher import positive_to_dustbin_margin_loss
            m, n = scores.shape
            logits = torch.cat((torch.cat((scores, dustbin.expand(m, 1)), 1), dustbin.expand(1, n+1)), 0)
            auxiliary = positive_to_dustbin_margin_loss(
                {'candidate_assignment_logits': logits[None]}, positive[None],
                torch.ones(1, dtype=torch.bool, device=scores.device), margin=margin)
        elif objective == 'known_cells':
            relative = scores-dustbin
            terms = []
            if positive.any(): terms.append(F.softplus(-relative[positive]).mean())
            if negative.any(): terms.append(F.softplus(relative[negative]).mean())
            if terms: auxiliary = torch.stack(terms).mean()
    terms = ([assignment] if assignment is not None else [])
    if auxiliary is not None and auxiliary_weight: terms.append(auxiliary_weight*auxiliary)
    return (sum(terms) if terms else None), {'assignment': assignment, 'auxiliary': auxiliary}


def sampling_pools(rows, objective='assignment'):
    active = lambda r: any(t != -2 for t in r['target_a']+r['target_b'])
    if objective == 'assignment':
        eligible = [r for r in rows if active(r)]
        return ([r for r in eligible if any(t >= 0 for t in r['target_a']+r['target_b'])],
                [r for r in eligible if all(t < 0 for t in r['target_a']+r['target_b'])])
    has_positive = lambda r: any(1 in line for line in r['weak_cells'])
    return ([r for r in rows if has_positive(r)],
            [r for r in rows if not has_positive(r) and -1 in r['target_a']+r['target_b']])


def row_loss(matcher, views, row, objective='assignment', auxiliary_weight=.5, margin=.2):
    output = matcher(views[row['observation_a']], views[row['observation_b']])
    loss, components = pair_objective(output, row, objective, auxiliary_weight, margin)
    output['loss_components'] = components
    return loss, output


def evaluate_rows(cache, encoder, matcher, rows, device):
    encoder.eval(); matcher.eval()
    total, count, skipped = 0., 0, 0
    with torch.no_grad():
        # Bound evaluation memory independently of dataset size. This graph-free
        # cache is local to each batch and never used by the training path.
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset+8]
            views = encode_rows(cache, encoder, batch, device)
            for row in batch:
                loss, _ = row_loss(matcher, views, row)
                if loss is None: skipped += 1
                else: total += float(loss); count += 1
    return {'mean_loss': total/count if count else None, 'evaluated_pairs': count, 'skipped_pairs': skipped}


def train_pairs(cache, supervision, output, *, steps=200, batch_size=2, seed=123,
                device='cpu', resume=None, training_mode='smoke', encoder_checkpoint=None):
    if steps < 1 or batch_size < 1 or training_mode not in ('smoke', 'full'):
        raise ValueError('invalid training configuration')
    output = Path(output).resolve()
    if cache.root in output.parents or output == cache.root:
        raise ValueError('training output must be outside frozen source')
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError('use a new training output or resume')
    labels = read_supervision(supervision, cache)
    rows = [r for r in labels['pairs'] if r['official_split'] == 'train' and r['training_subset'] == 'fit'
            and any(t != -2 for t in r['target_a']+r['target_b'])]
    if not rows:
        raise ValueError('no eligible fit supervision; unknown labels are not negatives')
    houses = sorted({r['house_id'] for r in rows})
    dev_houses = {r['house_id'] for r in labels['pairs'] if r['training_subset'] == 'dev'}
    if set(houses) & dev_houses:
        raise ValueError('fit/dev house leakage')
    root = Path(__file__).resolve().parents[2]
    code = ['portallayout/models/opening_representation.py', 'portallayout/models/opening_pair_matcher.py',
            'portallayout/engine/opening_pair_training.py', 'portallayout/utils/opening_representation_adapter.py',
            'portallayout/utils/frozen_opening_cache.py', 'portallayout/utils/opening_pair_checkpoint.py']
    contract = {'source_snapshot': cache.snapshot, 'source_contract': cache.fingerprint,
                'supervision_hash': content_hash(supervision),
                'batch_size': batch_size, 'seed': seed, 'device': device, 'training_mode': training_mode,
                'lr': 1e-4, 'weight_decay': 1e-4, 'clip_grad': 1., 'fit_houses': houses,
                'code_sha256': {p: content_hash(root/p) for p in code},
                'encoder_initialization_sha256': content_hash(encoder_checkpoint) if encoder_checkpoint else None}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    history, start, skipped = [], 0, 0
    if resume:
        encoder, matcher, previous = load_checkpoint(resume, device)
        if previous['training_contract'] != contract:
            raise ValueError('resume training contract mismatch')
    else:
        encoder, matcher = OpeningRepresentationEncoder().to(device), OpeningPairMatcher().to(device)
        if encoder_checkpoint:
            initial = torch.load(str(encoder_checkpoint), map_location='cpu')
            encoder.load_state_dict(initial['state_dict'], strict=True)
    parameters = list(encoder.parameters())+list(matcher.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    if resume:
        optimizer.load_state_dict(previous['optimizer_state'])
        start, history, skipped = previous['step'], previous['history'], previous['skipped_batches']
        random.setstate(previous['python_rng']); np.random.set_state(previous['numpy_rng'])
        torch.set_rng_state(previous['torch_rng'])
        if device.startswith('cuda'): torch.cuda.set_rng_state_all(previous['cuda_rng'])
        before, initial_metrics = previous['initial_weights_hash'], previous['initial_metrics']
    else:
        before = weights_hash(encoder, matcher)
        initial_metrics = evaluate_rows(cache, encoder, matcher, rows, device)
    if steps < start:
        raise ValueError('requested step precedes checkpoint step')
    positive = [r for r in rows if any(t >= 0 for t in r['target_a']+r['target_b'])]
    negative = [r for r in rows if all(t < 0 for t in r['target_a']+r['target_b'])]
    elapsed_start = time.perf_counter(); gradient_report = {}
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for step in range(start, steps):
        encoder.train(); matcher.train(); optimizer.zero_grad(set_to_none=True)
        chosen = [random.choice((positive if j % 2 == 0 else negative) or rows) for j in range(batch_size)]
        views = encode_rows(cache, encoder, chosen, device)
        losses, unconverged = [], 0
        for row in chosen:
            loss, prediction = row_loss(matcher, views, row)
            unconverged += prediction['status'] == 'numerical_unconverged'
            if loss is not None: losses.append(loss)
        if not losses:
            skipped += 1
            history.append({'step': step+1, 'loss': None, 'skipped': True}); continue
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('nonfinite parameter gradient')
        if not gradient_report:
            for prefix, model in (('encoder', encoder), ('matcher', matcher)):
                for name, module in model.named_children():
                    gradient_report[prefix+'.'+name] = float(sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None))
            gradient_report['matcher.bin_score'] = float(matcher.bin_score.grad.abs())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        history.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                        'valid_pairs': len(losses), 'unconverged_pairs': unconverged})
        if (step+1) % 25 == 0: print('step {}/{} loss={:.4f}'.format(step+1, steps, float(loss)), flush=True)
    final_metrics = evaluate_rows(cache, encoder, matcher, rows, device)
    after = weights_hash(encoder, matcher)
    count = cache.check_unchanged()
    status = 'smoke_trained' if training_mode == 'smoke' else 'trained'
    payload = checkpoint_payload(encoder, matcher, training_status=status, calibration=None,
        step=steps, optimizer_state=optimizer.state_dict(), training_contract=contract,
        history=history, skipped_batches=skipped, initial_weights_hash=before, initial_metrics=initial_metrics,
        python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
    save_checkpoint(output/'checkpoint.pt', payload)
    report = {'training_status': status, 'steps': steps, 'initial': initial_metrics, 'final': final_metrics,
              'parameters_changed': before != after, 'initial_weights_hash': before, 'weights_hash': after,
              'gradient_norms_first_step': gradient_report, 'source_files_verified': count,
              'fit_houses': houses, 'fit_pairs': len(rows), 'positive_pairs': len(positive),
              'negative_only_pairs': len(negative), 'skipped_batches': skipped,
              'supervision': labels['statistics'], 'elapsed_seconds': time.perf_counter()-elapsed_start,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
              'loss_decreased': final_metrics['mean_loss'] is not None and initial_metrics['mean_loss'] is not None and final_metrics['mean_loss'] < initial_metrics['mean_loss'],
              'generalization_evaluated': False}
    report['status'] = 'passed' if report['parameters_changed'] and report['loss_decreased'] else 'diagnostic_failed'
    write_json_atomic(output/'training_report.json', report)
    write_json_atomic(output/'history.json', history)
    return report


def calibrate_pairs(cache, supervision, checkpoint, output, device='cpu', min_precision=.95):
    """Calibrate on reviewed dev rows only; weak metrics cannot certify vision."""
    if not 0 < min_precision <= 1:
        raise ValueError('precision target must lie in (0,1]')
    encoder, matcher, payload = load_checkpoint(checkpoint, device)
    if payload['training_status'] != 'trained':
        raise ValueError('smoke or initialized checkpoints cannot be calibrated for production')
    labels = read_supervision(supervision, cache)
    rows = [r for r in labels['pairs'] if r['official_split'] == 'train' and r['training_subset'] == 'dev'
            and any(t != -2 for t in r['review_target_a']+r['review_target_b'])]
    if not rows or set(r['house_id'] for r in rows) & set(payload['training_contract']['fit_houses']):
        raise ValueError('calibration needs reviewed dev houses disjoint from fit')
    encoder.eval(); matcher.eval(); predictions = []
    positive_count = negative_rows = 0
    with torch.no_grad():
        views = encode_rows(cache, encoder, rows, device)
        for row in rows:
            result = matcher(views[row['observation_a']], views[row['observation_b']])
            target = row['review_target_a']
            positive_count += sum(t >= 0 for t in target)
            negative_rows += sum(t == -1 for t in target)
            for match in mutual_matches(result, 0.):
                truth = target[match['index_a']]
                if truth != -2:
                    predictions.append((match['confidence'], truth == match['index_b']))
    if not positive_count or not negative_rows or not predictions:
        raise ValueError('calibration requires reviewed positives, unmatched rows and predictions')
    options = []
    for threshold in sorted(set(c for c, _ in predictions)):
        accepted = [ok for c, ok in predictions if c >= threshold]
        precision = sum(accepted)/len(accepted)
        if precision >= min_precision and sum(accepted):
            options.append((sum(accepted)/positive_count, -threshold, precision))
    if not options:
        raise ValueError('no threshold satisfies dev precision target')
    recall, negative_threshold, precision = max(options)
    payload['calibration'] = {'threshold': -negative_threshold, 'precision': precision, 'recall': recall,
        'source': 'human_reviewed_dev', 'weights_hash': payload['weights_hash'],
        'supervision_fingerprint': labels['fingerprint'], 'dev_houses': sorted({r['house_id'] for r in rows}),
        'min_precision': min_precision, 'reviewed_positive_count': positive_count}
    if Path(output).exists(): raise ValueError('calibration output must be new')
    save_checkpoint(output, payload); cache.check_unchanged()
    return payload['calibration']
