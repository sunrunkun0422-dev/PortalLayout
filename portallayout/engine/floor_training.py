"""Staged, house-isolated training using frozen frontend features."""
from collections import OrderedDict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from portallayout.models.floor_auxiliary import PoseHypothesisVerifier, SameSpaceAssociator, FEATURE_VERSION, SUPPORTED_FEATURE_VERSIONS
from portallayout.models.opening_representation import OpeningRepresentationEncoder
from portallayout.models.opening_pair_matcher import OpeningPairMatcher
from portallayout.utils.opening_pair_checkpoint import checkpoint_payload, save_checkpoint, load_checkpoint, weights_hash
from portallayout.engine.opening_pair_training import read_supervision, encode_rows, row_loss, evaluate_rows, sampling_pools, PAIR_OBJECTIVES
from portallayout.geometry.floor_auxiliary_features import feature_tensors
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def _new_output(output, resume=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and resume is None: raise ValueError('use a new output or explicit resume')
    output.mkdir(parents=True, exist_ok=True)
    return output


def _state_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _validate_update(model):
    parameters = list(model.parameters())
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters): raise ValueError('nonfinite gradient')
    return float(sum(p.grad.norm().item() for p in parameters if p.grad is not None))


def _rng_state(rng):
    return {'python': rng.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _restore_rng(rng, state):
    rng.setstate(state['python']); torch.set_rng_state(state['torch'])
    if torch.cuda.is_initialized() and state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def train_floor_pairs(cache, supervision, protocol, output, *, steps=20000, batch_size=2,
                      seed=123, device='cpu', mode='full', validation_interval=500, patience=5,
                      validation_limit=2048, resume=None, initialization=None, matcher_type='new',
                      objective='assignment', auxiliary_weight=.5, margin=.2,
                      selection_metric='assignment_loss', dataset_root=None, protocol_seed=123):
    if mode not in ('full', 'smoke') or steps < 1 or batch_size < 1 or validation_interval < 1:
        raise ValueError('invalid training configuration')
    if mode == 'full' and steps < validation_interval: raise ValueError('full mode must include at least one development validation')
    if objective not in PAIR_OBJECTIVES or selection_metric not in ('assignment_loss', 'cross_room_ap'):
        raise ValueError('invalid objective or selection metric')
    if not np.isfinite([auxiliary_weight, margin]).all() or auxiliary_weight < 0 or margin < 0:
        raise ValueError('invalid auxiliary loss settings')
    output = _new_output(output, resume)
    if output == cache.root or cache.root in output.parents: raise ValueError('training output overlaps frozen cache')
    labels = read_supervision(supervision, cache)
    fit = [r for r in labels['pairs'] if r['house_id'] in protocol['fit']]
    positive, negative = sampling_pools(fit, objective)
    rows = sorted(positive+negative, key=lambda r: r['pair_id'])
    dev = [r for r in labels['pairs'] if r['house_id'] in protocol['selection'] and any(t != -2 for t in r['target_a']+r['target_b'])]
    dev = sorted(dev, key=lambda r: fingerprint([seed, r['pair_id']]))[:validation_limit]
    scopes, annotation_hashes = {}, {}
    if selection_metric == 'cross_room_ap':
        if dataset_root is None: raise ValueError('cross-room selection requires offline membership annotations')
        from portallayout.evaluation.opening_matching import room_scopes, evaluate_matching, selection_key
        dev = [r for r in labels['pairs'] if r['house_id'] in protocol['selection']]
        scopes, annotation_hashes = room_scopes(rows+dev, dataset_root)
        if not any(scopes[r['pair_id']] == 'cross_room' and any(1 in x for x in r['weak_cells']) for r in dev):
            raise ValueError('selection requires cross-room weak positives')
    if not rows or (mode == 'full' and not dev): raise ValueError('fit supervision and independent selection houses are required')
    if set(r['house_id'] for r in rows)&set(r['house_id'] for r in dev): raise ValueError('fit/dev leakage')
    contract = {'source_snapshot': cache.snapshot, 'source_contract': cache.fingerprint, 'supervision_sha256': content_hash(supervision),
        'house_protocol': protocol, 'batch_size': batch_size, 'seed': seed, 'mode': mode, 'device': device,
        'lr': 1e-4, 'weight_decay': 1e-4, 'clip_grad': 1., 'validation_interval': validation_interval,
        'validation_limit': validation_limit, 'patience': patience, 'fit_houses': sorted({r['house_id'] for r in rows}),
        'initialization_sha256': content_hash(initialization) if initialization else None,
        'objective': objective, 'auxiliary_weight': auxiliary_weight, 'margin': margin,
        'selection_metric': selection_metric, 'protocol_seed': protocol_seed,
        'selection_annotation_sha256': annotation_hashes}
    torch.manual_seed(seed); rng = random.Random(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if resume:
        encoder, matcher, previous = load_checkpoint(resume, device)
        old_contract = dict(previous['training_contract'])
        for key, default in dict(objective='assignment', auxiliary_weight=.5, margin=.2,
                                 selection_metric='assignment_loss', protocol_seed=123,
                                 selection_annotation_sha256={}).items(): old_contract.setdefault(key, default)
        if old_contract != contract: raise ValueError('resume contract differs')
    elif initialization:
        encoder, matcher, _ = load_checkpoint(initialization, device); previous = None
    else:
        encoder, matcher = OpeningRepresentationEncoder().to(device), OpeningPairMatcher().to(device); previous = None
    if matcher_type not in ('new', 'legacy'): raise ValueError('unknown matcher type')
    if getattr(matcher, 'matcher_type', 'new') != matcher_type:
        if resume: raise ValueError('resume matcher architecture differs')
        if matcher_type == 'legacy':
            from portallayout.models.legacy_floor_matcher import LegacyFloorMatcher, LegacyFloorConfig
            matcher = LegacyFloorMatcher(LegacyFloorConfig(feature_dim=encoder.config.feature_1d_dim)).to(device)
        else: matcher = OpeningPairMatcher().to(device)
    train_encoder = getattr(matcher, 'trains_encoder', True)
    encoder.requires_grad_(train_encoder)
    params = list(encoder.parameters())+list(matcher.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    probe = sorted(rows, key=lambda r: fingerprint([seed, r['pair_id']]))[:256]
    start, history, best, stale = 0, [], None, 0
    initial_hash = weights_hash(encoder, matcher)
    initial = evaluate_rows(cache, encoder, matcher, probe, device)
    initial_quality = None
    if selection_metric == 'cross_room_ap':
        _, initial_quality = evaluate_matching(cache, encoder, matcher, probe, scopes, device,
                                              objective, auxiliary_weight, margin)
    gradient_report = {}; best_state = None
    best_key = None; best_step = None; best_quality = None; assignment_best = None; assignment_step = None
    if previous:
        optimizer.load_state_dict(previous['optimizer_state']); _restore_rng(rng, previous['rng'])
        start, history, best, stale = previous['step'], previous['history'], previous['best_loss'], previous['stale']
        initial, initial_hash = previous['initial_metrics'], previous['initial_weights_hash']
        gradient_report = previous['gradient_report']; best_state = previous['best_state']
        best_key = previous.get('best_key'); best_step = previous.get('best_step')
        best_quality = previous.get('best_quality'); assignment_best = previous.get('assignment_best')
        assignment_step = previous.get('assignment_step'); initial_quality = previous.get('initial_quality', initial_quality)
    if steps < start: raise ValueError('resume step precedes saved step')
    if not positive or not negative: raise ValueError('fit requires both positive and proven-unmatched supervision')
    pools = []
    for values in (positive, negative):
        pool = {}
        for r in values: pool.setdefault(r['house_id'], []).append(r)
        pools.append(pool)
    timer = time.perf_counter(); skipped = previous.get('skipped_batches', 0) if previous else 0
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    completed_step = start
    for step in range(start, start if stale >= patience else steps):
        chosen = []
        for j in range(batch_size):
            pool = pools[(step*batch_size+j)%2]; house = rng.choice(sorted(pool)); chosen.append(rng.choice(pool[house]))
        encoder.train(); matcher.train(); optimizer.zero_grad(set_to_none=True)
        views = encode_rows(cache, encoder, chosen, device)
        evaluated = [row_loss(matcher, views, row, objective, auxiliary_weight, margin) for row in chosen]
        losses = [x for x, _ in evaluated if x is not None]
        completed_step = step+1
        if not losses:
            skipped += 1; history.append({'step': completed_step, 'loss': None, 'skipped': True}); continue
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite assignment loss')
        loss.backward()
        grads = {'encoder': _validate_update(encoder), 'matcher': _validate_update(matcher)}
        if not gradient_report: gradient_report = grads
        torch.nn.utils.clip_grad_norm_(params, 1.); optimizer.step()
        history.append({'step': completed_step, 'loss': float(loss.detach()), 'valid_pairs': len(losses)})
        for name in ('assignment', 'auxiliary'):
            values = [out['loss_components'][name] for _, out in evaluated if out['loss_components'][name] is not None]
            history[-1][name+'_loss'] = float(torch.stack(values).mean().detach()) if values else None
        if completed_step % 100 == 0: print('floor pair step={} loss={:.5f}'.format(completed_step, float(loss.detach())), flush=True)
        if completed_step % validation_interval == 0 or completed_step == steps:
            if selection_metric == 'cross_room_ap':
                _, validation = evaluate_matching(cache, encoder, matcher, dev, scopes, device,
                                                   objective, auxiliary_weight, margin)
                value = validation['assignment_loss']; key = selection_key(validation)
                improved = best_key is None or key > tuple(best_key)
            else:
                validation = evaluate_rows(cache, encoder, matcher, dev or probe, device)
                value = validation['mean_loss']; key = None
                improved = value is not None and (best is None or value < best-1e-4)
            history[-1]['selection'] = validation
            if value is not None and (assignment_best is None or value < assignment_best-1e-4):
                assignment_best, assignment_step = value, completed_step
                save_checkpoint(output/'best_assignment.pt', checkpoint_payload(encoder, matcher,
                    training_status='smoke_trained' if mode == 'smoke' else 'trained', calibration=None,
                    training_contract=contract, step=completed_step, selected_by='assignment_loss'))
            if improved:
                best, stale, best_key, best_step, best_quality = value, 0, key, completed_step, validation
                best_state = {name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                              for name, model in (('encoder', encoder), ('matcher', matcher))}
            else: stale += 1
            payload = checkpoint_payload(encoder, matcher, training_status='smoke_trained' if mode == 'smoke' else 'trained',
                calibration=None, training_contract=contract, optimizer_state=optimizer.state_dict(), step=completed_step,
                history=history, rng=_rng_state(rng), best_loss=best, best_state=best_state, stale=stale,
                gradient_report=gradient_report, initial_metrics=initial, initial_weights_hash=initial_hash,
                best_key=best_key, best_step=best_step, best_quality=best_quality, assignment_best=assignment_best,
                assignment_step=assignment_step, initial_quality=initial_quality, skipped_batches=skipped)
            save_checkpoint(output/'last.pt', payload)
            print('selection step={} metric={} value={} stale={}'.format(completed_step, selection_metric,
                  key if key is not None else value, stale), flush=True)
            if stale >= patience: break
    if best_state is not None:
        encoder.load_state_dict(best_state['encoder']); matcher.load_state_dict(best_state['matcher'])
    final = evaluate_rows(cache, encoder, matcher, probe, device)
    final_quality = None
    if selection_metric == 'cross_room_ap':
        _, final_quality = evaluate_matching(cache, encoder, matcher, probe, scopes, device,
                                            objective, auxiliary_weight, margin)
    changed = weights_hash(encoder, matcher) != initial_hash
    required_gradients = ['encoder', 'matcher'] if train_encoder else ['matcher']
    if not changed or not all(gradient_report.get(k, 0) > 0 for k in required_gradients): raise ValueError('trainable modules did not receive finite updates')
    payload = checkpoint_payload(encoder, matcher, training_status='smoke_trained' if mode == 'smoke' else 'trained',
        calibration=None, training_contract=contract, step=completed_step, selection_loss=best,
        selected_by=selection_metric if dev else 'fit_probe_smoke_only', selected_step=best_step, selection_metrics=best_quality)
    save_checkpoint(output/'checkpoint.pt', payload)
    report = {'status': 'passed' if final['mean_loss'] < initial['mean_loss'] else 'diagnostic_loss_not_decreased',
        'training_status': payload['training_status'], 'steps': completed_step, 'initial_fit_probe': initial,
        'final_fit_probe': final, 'selection_loss': best, 'selection_pairs': len(dev),
        'selected_step': best_step, 'selection_metric': selection_metric, 'selection_metrics': best_quality,
        'best_assignment_loss': assignment_best, 'best_assignment_step': assignment_step,
        'initial_fit_quality': initial_quality, 'final_fit_quality': final_quality,
        'objective': objective, 'sampler_positive_pairs': len(positive), 'sampler_unmatched_row_pairs': len(negative),
        'fit_houses': contract['fit_houses'], 'selection_houses': sorted({r['house_id'] for r in dev}),
        'parameters_changed': changed, 'matcher_type': matcher_type, 'encoder_frozen': not train_encoder,
        'gradient_norms_first_step': gradient_report, 'skipped_batches': skipped,
        'elapsed_seconds': time.perf_counter()-timer, 'source_files_verified': cache.check_unchanged(),
        'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
        'final_benchmark_evaluated': False}
    if any(content_hash(path) != h for path, h in annotation_hashes.items()):
        raise ValueError('offline selection annotations changed during training')
    write_json_atomic(output/'training_report.json', report); write_json_atomic(output/'history.json', history)
    return report


class AuxiliaryFeatureDataset:
    def __init__(self, supervision):
        self.path = Path(supervision).resolve(); self.data = json.loads(self.path.read_text())
        claimed = self.data.pop('fingerprint')
        if fingerprint(self.data) != claimed: raise ValueError('auxiliary supervision fingerprint mismatch')
        self.data['fingerprint'] = claimed
        if self.data['feature_version'] not in SUPPORTED_FEATURE_VERSIONS: raise ValueError('auxiliary feature version mismatch')
        protocol = self.data['house_protocol']
        house_roles = {}
        for role in ('fit', 'selection', 'calibration'):
            for house in protocol[role]:
                if house in house_roles: raise ValueError('auxiliary house splits overlap')
                house_roles[house] = role
        for row in self.data['records']+self.data.get('opening_pairs', []):
            if house_roles.get(row['house_id']) != row['role']:
                raise ValueError('auxiliary row role disagrees with house protocol')
        record_ids = [row['record_id'] for row in self.data['records']]
        if len(record_ids) != len(set(record_ids)): raise ValueError('duplicate auxiliary record identity')
        self.cache = OrderedDict()

    def load(self, row):
        from portallayout.datasets.zind_floor_dataset import safe_path
        key = row['record_id']
        if key not in self.cache:
            path = safe_path(self.path.parent, row['feature_file'])
            if content_hash(path) != row['feature_sha256']: raise ValueError('auxiliary feature checksum mismatch')
            with np.load(path, allow_pickle=False) as value:
                self.cache[key] = {k: value[k].astype(np.float32) for k in ('raster', 'visual', 'geometry')}
        self.cache.move_to_end(key)
        while len(self.cache) > 128: self.cache.popitem(last=False)
        return self.cache[key]


def predict_auxiliary(model, dataset, rows, device='cpu', batch_size=32):
    model.eval(); result = []
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            result.extend(model(*feature_tensors([dataset.load(r) for r in batch], device)).sigmoid().cpu().tolist())
    return result


def train_auxiliaries(supervision, pair_checkpoint, output, *, steps=5000, batch_size=32,
                      seed=123, device='cpu', mode='full', validation_interval=500, patience=5, resume=None,
                      tasks=('pose', 'room'), initialization=None, selection_metric='bce'):
    if steps < 1 or batch_size < 1 or mode not in ('full', 'smoke'): raise ValueError('invalid auxiliary training config')
    if mode == 'full' and steps < validation_interval: raise ValueError('full mode requires development validation')
    if selection_metric not in ('bce', 'cross_room_ap'): raise ValueError('invalid pose selection metric')
    if selection_metric == 'cross_room_ap' and tuple(tasks) != ('pose',): raise ValueError('pose AP requires pose-only training')
    output = _new_output(output, resume); dataset = AuxiliaryFeatureDataset(supervision)
    if dataset.data['pair_checkpoint_sha256'] != content_hash(pair_checkpoint): raise ValueError('pair/auxiliary feature fingerprint mismatch')
    contract = {'supervision_sha256': content_hash(supervision), 'pair_checkpoint_sha256': content_hash(pair_checkpoint),
        'seed': seed, 'device': device, 'batch_size': batch_size, 'mode': mode, 'lr': 1e-4, 'weight_decay': 1e-4,
        'clip_grad': 1., 'validation_interval': validation_interval, 'patience': patience,
        'house_protocol': dataset.data['house_protocol']}
    if selection_metric != 'bce':
        contract.update(selection_metric=selection_metric, feature_version=dataset.data['feature_version'], sampler='balanced_class_house.v1')
    if dataset.data.get('pose_geometry') is not None:
        contract['pose_geometry'] = dataset.data['pose_geometry']
    torch.manual_seed(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    models = {'pose': PoseHypothesisVerifier().to(device), 'room': SameSpaceAssociator().to(device)}
    states, reports = {}, {task: {'status': 'untrained'} for task in models}
    if not tasks or not set(tasks) <= {'pose','room'}: raise ValueError('invalid auxiliary tasks')
    if tuple(tasks) != ('pose','room'):
        contract['tasks'] = list(tasks)
    if initialization:
        initial_state = torch.load(initialization, map_location='cpu')
        if initial_state.get('feature_version') != dataset.data['feature_version'] or initial_state.get('pair_checkpoint_sha256') != content_hash(pair_checkpoint):
            raise ValueError('auxiliary initialization contract mismatch')
        if initial_state.get('training_contract', {}).get('mode') != 'full':
            raise ValueError('auxiliary initialization must come from full training')
        if initial_state['training_contract']['house_protocol'] != dataset.data['house_protocol']:
            raise ValueError('auxiliary initialization house protocol mismatch')
        for task, model in models.items(): model.load_state_dict(initial_state[task+'_state'])
        reports = dict(initial_state['reports'])
        contract.update(tasks=list(tasks), initialization_sha256=content_hash(initialization))
    if dataset.data.get('room_feature_source') == 'predicted_pgo_components':
        if tuple(tasks) != ('room',) or _state_hash(models['pose']) != dataset.data['room_pose_model_hash']:
            raise ValueError('post-PGO room training must freeze the source pose verifier')
    previous = torch.load(resume, map_location='cpu') if resume else None
    if previous and previous['training_contract'] != contract: raise ValueError('auxiliary resume contract differs')
    if previous:
        for task, model in models.items(): model.load_state_dict(previous[task+'_state']); states = previous['stages']
    start_time = time.perf_counter()
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for task_index, (task, model) in enumerate(models.items()):
        if task not in tasks: continue
        fit = [r for r in dataset.data['records'] if r['task'] == task and r['role'] == 'fit' and r['target'] >= 0]
        dev = [r for r in dataset.data['records'] if r['task'] == task and r['role'] == 'selection' and r['target'] >= 0]
        dev = sorted(dev, key=lambda r: fingerprint([seed, r['record_id']]))
        if {r['target'] for r in fit} != {0, 1} or (mode == 'full' and {r['target'] for r in dev} != {0, 1}):
            reports[task] = {'status': 'insufficient_supervision', 'fit': len(fit), 'selection': len(dev)}; continue
        if selection_metric == 'cross_room_ap' and mode == 'full' and not any(r['target'] == 1 and r.get('relationship') == 'cross_room' for r in dev):
            reports[task] = {'status': 'no_cross_room_selection_positives'}; continue
        if {r['house_id'] for r in fit}&{r['house_id'] for r in dev}: raise ValueError('auxiliary house leakage')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        rng = random.Random(seed+task_index); stage = states.get(task, {})
        history, first, best, stale, best_state = stage.get('history', []), stage.get('step', 0), stage.get('best_loss'), stage.get('stale', 0), stage.get('best_state')
        best_key = tuple(stage['best_key']) if stage.get('best_key') is not None else None
        best_step = stage.get('best_step'); best_bce_step = stage.get('best_bce_step')
        best_bce = stage.get('best_bce'); best_bce_state = stage.get('best_bce_state')
        initial_hash = stage.get('initial_hash', _state_hash(model)); gradient = stage.get('gradient_norm', 0.)
        if stage:
            if 'last_model_state' in stage:
                model.load_state_dict(stage['last_model_state'])
            elif stage.get('finished') and stale < patience and steps > first:
                raise ValueError('legacy finished stage lacks optimizer-aligned weights; restart auxiliary training')
            optimizer.load_state_dict(stage['optimizer']); _restore_rng(rng, stage['rng'])
        if steps < first: raise ValueError('resume requested before saved auxiliary step')
        probe = sorted(fit, key=lambda r: fingerprint(r['record_id']))[:256]
        def evaluate(rows):
            probability = np.clip(predict_auxiliary(model, dataset, rows, device), 1e-7, 1-1e-7)
            targets = np.array([r['target'] for r in rows])
            return float(np.mean(-targets*np.log(probability)-(1-targets)*np.log(1-probability)))
        initial = stage.get('initial_fit_loss', evaluate(probe)); completed = first
        pools = {}
        for r in fit: pools.setdefault(r['target'], {}).setdefault(r['house_id'], []).append(r)
        for step in range(first, first if stale >= patience else steps):
            batch = []
            for j in range(batch_size):
                pool = pools[(j+step)%2]; house = rng.choice(sorted(pool)); batch.append(rng.choice(pool[house]))
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(*feature_tensors([dataset.load(r) for r in batch], device))
            target = torch.tensor([r['target'] for r in batch], device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss): raise ValueError('nonfinite auxiliary loss')
            loss.backward(); g = _validate_update(model); gradient = gradient or g
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); completed = step+1
            history.append({'step': completed, 'loss': float(loss.detach())})
            if completed % 100 == 0: print('{} step={} loss={:.5f}'.format(task, completed, float(loss.detach())), flush=True)
            if completed % validation_interval == 0 or completed == steps:
                selection = evaluate(dev or probe); history[-1]['selection_loss'] = selection
                if best_bce is None or selection < best_bce:
                    best_bce_step = completed
                    best_bce = selection; best_bce_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                key = (-selection,)
                if selection_metric == 'cross_room_ap':
                    from portallayout.evaluation.pose_verifier import quality
                    values = predict_auxiliary(model, dataset, dev or probe, device)
                    q = quality([dict(r, probability=float(v)) for r, v in zip(dev or probe, values)])
                    history[-1]['quality'] = q
                    key = (q['cross_room_house_macro_ap'] if q['cross_room_house_macro_ap'] is not None else -1.,
                           q['overall_ap'] if q['overall_ap'] is not None else -1., -q['balanced_bce'])
                improved = (best is None or selection < best-1e-4) if selection_metric == 'bce' else (best_key is None or key > best_key)
                if improved:
                    best_key = key; best_step = completed
                    best, stale = selection, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else: stale += 1
                states[task] = {'step': completed, 'optimizer': optimizer.state_dict(), 'rng': _rng_state(rng),
                    'best_step': best_step, 'best_bce_step': best_bce_step, 'best_key': best_key, 'best_bce': best_bce, 'best_bce_state': best_bce_state, 'best_loss': best, 'best_state': best_state, 'stale': stale, 'history': history,
                    'initial_hash': initial_hash, 'gradient_norm': gradient, 'initial_fit_loss': initial,
                    'last_model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                save_checkpoint(output/'last.pt', dict(training_contract=contract, stages=states,
                    **{t+'_state': m.state_dict() for t, m in models.items()}))
                if stale >= patience: break
        if task in states: states[task]['finished'] = True
        if best_state is not None: model.load_state_dict(best_state)
        final = evaluate(probe); changed = _state_hash(model) != initial_hash
        if not changed or gradient <= 0: raise ValueError('auxiliary parameters did not receive finite updates')
        reports[task] = {'status': 'passed' if final < initial else 'diagnostic_loss_not_decreased',
            'selected_step': best_step, 'best_bce_step': best_bce_step, 'selection_metric': selection_metric, 'selection_key': best_key, 'steps': completed, 'fit': len(fit), 'selection': len(dev), 'initial_fit_loss': initial,
            'final_fit_loss': final, 'selection_loss': best, 'parameters_changed': changed,
            'gradient_norm_first_step': gradient, 'fit_houses': sorted({r['house_id'] for r in fit}),
            'selection_houses': sorted({r['house_id'] for r in dev})}
    successful = all(r['status'] in ('passed', 'diagnostic_loss_not_decreased') for t, r in reports.items() if t in tasks)
    status = ('smoke_trained' if mode == 'smoke' else 'trained') if successful else 'incomplete_training'
    save_checkpoint(output/'checkpoint.pt', {'schema_version': 'floor-auxiliary-checkpoint.v1',
        'feature_version': dataset.data['feature_version'], 'training_status': status, 'training_contract': contract,
        'pair_checkpoint_sha256': content_hash(pair_checkpoint),
        'pose_state': models['pose'].state_dict(), 'room_state': models['room'].state_dict(), 'reports': reports})
    if tuple(tasks) == ('pose',) and states.get('pose', {}).get('best_bce_state') is not None:
        payload = torch.load(output/'checkpoint.pt', map_location='cpu')
        payload['pose_state'] = states['pose']['best_bce_state']; payload['selection_variant'] = 'lowest_bce_diagnostic'
        save_checkpoint(output/'best_bce.pt', payload)
    report = {'training_status': status, 'stages': reports, 'elapsed_seconds': time.perf_counter()-start_time,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
              'final_benchmark_evaluated': False}
    write_json_atomic(output/'training_report.json', report)
    for task in states: write_json_atomic(output/(task+'_history.json'), states[task]['history'])
    return report
