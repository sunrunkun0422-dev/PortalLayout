#!/usr/bin/env python3
"""Train, calibrate, reconstruct, or independently evaluate whole floors."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'calibrate', 'infer', 'evaluate'):
        q = sub.add_parser(name); q.add_argument('--output', required=True)
        q.add_argument('--device', default='auto'); q.add_argument('--torch-threads', type=int, default=2)
        if name in ('prepare', 'train', 'infer'): q.add_argument('--prediction-cache')
        if name in ('prepare', 'train', 'calibrate', 'evaluate'): q.add_argument('--dataset')
        if name in ('prepare', 'train', 'calibrate'): q.add_argument('--house-protocol')
        if name in ('prepare', 'train', 'calibrate', 'infer'): q.add_argument('--pair-checkpoint')
        if name in ('train', 'calibrate'): q.add_argument('--supervision')
        if name == 'prepare':
            q.add_argument('--stage', choices=('opening', 'audit-opening', 'auxiliary', 'room-pgo'), default='opening')
            q.add_argument('--supervision'); q.add_argument('--predictions'); q.add_argument('--bundle')
        if name == 'train':
            q.add_argument('--stage', choices=('pairs', 'auxiliary', 'all'), default='all')
            q.add_argument('--matcher-type', choices=('new', 'legacy'), default='new')
            q.add_argument('--auxiliary-task', choices=('both', 'pose', 'room'), default='both')
            q.add_argument('--auxiliary-initialization')
            q.add_argument('--auxiliary-selection-metric', choices=('bce','cross_room_ap'), default='bce')
            q.add_argument('--mode', choices=('full', 'smoke'), default='full')
            q.add_argument('--pair-steps', type=int, default=20000); q.add_argument('--auxiliary-steps', type=int, default=5000)
            q.add_argument('--pair-batch-size', type=int, default=2); q.add_argument('--auxiliary-batch-size', type=int, default=32)
            q.add_argument('--seed', type=int, default=123); q.add_argument('--resume')
            q.add_argument('--protocol-seed', type=int, default=123)
            q.add_argument('--pair-objective', choices=('assignment','positive_margin','known_cells'), default='assignment')
            q.add_argument('--auxiliary-weight', type=float, default=.5); q.add_argument('--positive-margin', type=float, default=.2)
            q.add_argument('--selection-metric', choices=('assignment_loss','cross_room_ap'), default='assignment_loss')
            q.add_argument('--validation-interval', type=int, default=500); q.add_argument('--patience', type=int, default=5)
        if name == 'calibrate':
            q.add_argument('--auxiliary-checkpoint', required=True); q.add_argument('--minimum-houses', type=int, default=5)
            q.add_argument('--opening-report')
        if name == 'infer':
            q.add_argument('--inference-mode', choices=('strict', 'exploratory'), default='strict')
            q.add_argument('--opening-threshold', type=float, default=.5)
            q.add_argument('--room-association', choices=('learned', 'geometry', 'none'), default='learned')
            q.add_argument('--pair-prediction-cache')
            q.add_argument('--pose-policy', choices=('geometry','verifier_rank','verifier_filter'), default='geometry')
            q.add_argument('--pose-workpoint')
            q.add_argument('--pose-prediction-cache')
            q.add_argument('--pose-geometry', choices=('dominant_wall','raw_depth','local_m2'))
            q.add_argument('--scale-shrinkage', type=float, choices=(1.,.5,0.), default=1.)
            q.add_argument('--bundle'); q.add_argument('--auxiliary-checkpoint'); q.add_argument('--calibration')
            q.add_argument('--manifest'); q.add_argument('--raw-root'); q.add_argument('--frontend-cache-output')
            q.add_argument('--frontend-config', default=str(ROOT/'configs/research1_floor_frontend.json'))
            q.add_argument('--pose-opt', choices=('none', 'robust'), default='robust')
            q.add_argument('--no-auxiliary-verification', action='store_true'); q.add_argument('--no-room-fusion', action='store_true')
            q.add_argument('--floor-uids', nargs='+')
        if name == 'evaluate':
            q.add_argument('--predictions', required=True); q.add_argument('--scale-track', choices=('S', 'M'), default='S')
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    from portallayout.utils.room_feature_cache import write_json_atomic
    output = Path(args.output).resolve()
    if args.command != 'train' or not args.resume:
        if output.exists() and any(output.iterdir()): p.error('output must be a new/empty directory')
    output.mkdir(parents=True, exist_ok=True)
    def require(*names):
        for name in names:
            if not getattr(args, name, None): p.error('--'+name.replace('_', '-')+' is required for this operation')
    if args.command == 'prepare':
        require('prediction_cache', 'dataset')
        from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
        from portallayout.datasets.floor_auxiliary_supervision import (house_protocol, validate_cache_house_roles,
                                                                      validate_auxiliary_for_training)
        protocol = house_protocol(args.dataset, protocol_path=args.house_protocol) if args.house_protocol else None
        if args.stage == 'audit-opening' and protocol is None: p.error('audit-opening requires --house-protocol')
        cache = FrozenOpeningCache(args.prediction_cache)
        if protocol:
            validate_cache_house_roles(cache, args.dataset, protocol,
                {'audit'} if args.stage == 'audit-opening' else {'fit', 'selection', 'calibration'},
                Path(args.house_protocol).resolve().parent/'cohort.json')
        if args.stage in ('opening', 'audit-opening'):
            from portallayout.datasets.opening_pair_supervision import build_weak_supervision
            result = build_weak_supervision(cache, args.dataset, output/'opening_weak.json',
                                            protocol=protocol, audit_only=args.stage == 'audit-opening')['statistics']
        elif args.stage == 'room-pgo':
            require('supervision', 'predictions', 'bundle')
            if protocol: validate_auxiliary_for_training(args.supervision, protocol)
            from portallayout.datasets.floor_room_supervision import build_post_pgo_room_supervision
            result = build_post_pgo_room_supervision(cache, args.dataset, args.supervision,
                args.predictions, args.bundle, output, device)['statistics']
        else:
            require('pair_checkpoint')
            from portallayout.datasets.floor_auxiliary_supervision import build_auxiliary_supervision
            result = build_auxiliary_supervision(cache, args.dataset, args.pair_checkpoint, output, device,
                                                 protocol=protocol)['statistics']
    elif args.command == 'train':
        from portallayout.engine.floor_training import train_floor_pairs, train_auxiliaries
        from portallayout.datasets.floor_auxiliary_supervision import (house_protocol, build_auxiliary_supervision,
            validate_cache_house_roles, validate_supervision_house_roles, validate_auxiliary_for_training)
        from portallayout.datasets.opening_pair_supervision import build_weak_supervision
        if args.stage == 'all' and args.resume: p.error('resume requires an explicit pairs or auxiliary stage')
        if args.house_protocol: require('dataset')
        protocol = (house_protocol(args.dataset, args.protocol_seed, args.house_protocol)
                    if args.house_protocol else None)
        results = {}
        if args.stage in ('pairs', 'all'):
            require('prediction_cache', 'dataset')
            from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
            cache = FrozenOpeningCache(args.prediction_cache)
            if protocol is None: protocol = house_protocol(args.dataset, args.protocol_seed)
            if args.house_protocol:
                floor_houses = validate_cache_house_roles(cache, args.dataset, protocol,
                    {'fit', 'selection', 'calibration'}, Path(args.house_protocol).resolve().parent/'cohort.json')
            supervision = args.supervision or str(output/'opening_weak.json')
            if not args.supervision and not Path(supervision).exists():
                build_weak_supervision(cache, args.dataset, supervision, protocol=protocol if args.house_protocol else None)
            if args.house_protocol:
                from portallayout.engine.opening_pair_training import read_supervision
                labels = read_supervision(supervision, cache)
                if labels.get('house_protocol', protocol) != protocol:
                    raise ValueError('weak supervision and requested house protocols differ')
                validate_supervision_house_roles(labels, protocol, {'fit', 'selection', 'calibration'}, floor_houses, cache)
            target = output/'pairs' if args.stage == 'all' or not args.supervision else output
            results['pairs'] = train_floor_pairs(cache, supervision, protocol, target,
                steps=args.pair_steps, batch_size=args.pair_batch_size, seed=args.seed, device=device, mode=args.mode,
                validation_interval=args.validation_interval, patience=args.patience,
                resume=args.resume, initialization=args.pair_checkpoint, matcher_type=args.matcher_type,
                objective=args.pair_objective, auxiliary_weight=args.auxiliary_weight, margin=args.positive_margin,
                selection_metric=args.selection_metric, dataset_root=args.dataset, protocol_seed=args.protocol_seed)
            pair_checkpoint = target/'checkpoint.pt'
        else: require('pair_checkpoint'); pair_checkpoint = Path(args.pair_checkpoint)
        if args.stage in ('auxiliary', 'all'):
            if args.stage == 'all':
                auxiliary_root = output/'auxiliary_labels'
                build_auxiliary_supervision(cache, args.dataset, pair_checkpoint, auxiliary_root, device, args.protocol_seed,
                                            protocol=protocol if args.house_protocol else None)
                supervision = auxiliary_root/'supervision.json'; target = output/'auxiliary'
            else:
                require('supervision'); supervision = args.supervision; target = output
                if args.house_protocol: validate_auxiliary_for_training(supervision, protocol)
            results['auxiliary'] = train_auxiliaries(supervision, pair_checkpoint, target,
                steps=args.auxiliary_steps, batch_size=args.auxiliary_batch_size, seed=args.seed, device=device,
                mode=args.mode, validation_interval=args.validation_interval, patience=args.patience, resume=args.resume,
                tasks=('pose','room') if args.auxiliary_task == 'both' else (args.auxiliary_task,),
                initialization=args.auxiliary_initialization, selection_metric=args.auxiliary_selection_metric)
            write_json_atomic(output/'bundle_uncalibrated.json', {'pair_checkpoint': str(pair_checkpoint.resolve()),
                'auxiliary_checkpoint': str((target/'checkpoint.pt').resolve())})
        result = results
    elif args.command == 'calibrate':
        require('supervision', 'pair_checkpoint')
        if args.house_protocol:
            require('dataset')
            from portallayout.datasets.floor_auxiliary_supervision import house_protocol, validate_auxiliary_for_training
            validate_auxiliary_for_training(args.supervision, house_protocol(args.dataset, protocol_path=args.house_protocol))
        from portallayout.engine.floor_calibration import calibrate_floor_models
        result = calibrate_floor_models(args.supervision, args.pair_checkpoint, args.auxiliary_checkpoint,
                                       output/'calibration.json', device, args.minimum_houses, args.opening_report)
        write_json_atomic(output/'bundle.json', {'pair_checkpoint': str(Path(args.pair_checkpoint).resolve()),
            'auxiliary_checkpoint': str(Path(args.auxiliary_checkpoint).resolve()), 'calibration': str(output/'calibration.json')})
    elif args.command == 'infer':
        if bool(args.prediction_cache) == bool(args.manifest): p.error('choose exactly one of --prediction-cache and --manifest')
        if not args.bundle and not args.pair_checkpoint: p.error('--bundle or --pair-checkpoint is required')
        from portallayout.engine.floor_layout_pipeline import FloorModelBundle, estimate_floor, export_floor, failed_floor
        from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
        start = time.perf_counter()
        bundle = FloorModelBundle(args.bundle or {'pair_checkpoint': args.pair_checkpoint,
            'auxiliary_checkpoint': args.auxiliary_checkpoint, 'calibration': args.calibration}, device)
        model_seconds = time.perf_counter()-start; cache_start = time.perf_counter()
        cache = FrozenOpeningCache(args.prediction_cache) if args.prediction_cache else None
        inputs = cache.floors if cache else [json.loads(Path(args.manifest).read_text())]
        cache_seconds = time.perf_counter()-cache_start
        if args.floor_uids:
            inputs = [f for f in inputs if f['floor_uid'] in args.floor_uids]
            if set(args.floor_uids)-{f['floor_uid'] for f in inputs}: p.error('requested floor is absent')
        if len({f['floor_uid'] for f in inputs}) != len(inputs): p.error('one sample per floor is required; separate sample views')
        refs = []
        for floor in inputs:
            target = output/floor['floor_uid']
            cfg = {'device': device, 'pose_opt': args.pose_opt, 'auxiliary_verification': not args.no_auxiliary_verification,
                   'room_fusion': not args.no_room_fusion and args.room_association != 'none', 'pair_output': str(target/'pairs'),
                   'inference_mode': args.inference_mode, 'opening_threshold': args.opening_threshold,
                   'room_association': args.room_association, 'pair_prediction_cache': args.pair_prediction_cache,
                   'raw_root': args.raw_root, 'frontend_cache_output': args.frontend_cache_output,
                   'frontend_config': args.frontend_config, 'pose_policy': args.pose_policy,
                   'pose_workpoint': args.pose_workpoint, 'pose_prediction_cache': args.pose_prediction_cache}
            if args.pose_geometry is not None:
                if args.inference_mode != 'exploratory' or args.pose_policy == 'geometry':
                    p.error('--pose-geometry requires an exploratory verifier policy')
                cfg.update(pose_geometry=args.pose_geometry, scale_shrinkage=args.scale_shrinkage)
            try:
                prediction = estimate_floor(floor, bundle, cfg, cache)
            except (ValueError, RuntimeError, OSError) as error:
                # Persist failure instead of silently omitting a floor from evaluation.
                prediction = failed_floor(floor, error)
                if device.startswith('cuda'): torch.cuda.empty_cache()
            result_file = export_floor(prediction, target)
            refs.append({'floor_uid': floor['floor_uid'], 'result_file': result_file, 'status': prediction['status'],
                         'runtime': prediction['runtime'], 'components': [len(c['observation_ids']) for c in prediction['components']]})
            print('floor={} status={} components={}'.format(floor['floor_uid'], prediction['status'], refs[-1]['components']), flush=True)
        result = {'status': 'completed_with_failures' if any(r['status'] == 'runtime_failure' for r in refs) else 'complete',
            'floors': refs, 'model_load_seconds': model_seconds, 'cache_audit_seconds': cache_seconds,
            'total_including_load_export_seconds': time.perf_counter()-start,
            'source_files_verified': cache.check_unchanged() if cache else None, 'ground_truth_read': False}
        write_json_atomic(output/'run_report.json', result)
    else:
        require('dataset')
        from portallayout.evaluation.floor_metrics import load_floor_ground_truth, evaluate_floor, FloorMetricProtocol, write_metrics
        source = Path(args.predictions)
        if source.is_file(): paths = [source]
        elif (source/'run_report.json').is_file():
            paths = [Path(r['result_file']) for r in json.loads((source/'run_report.json').read_text())['floors']]
        else: paths = sorted(source.rglob('floor_result.json'))
        if not paths: p.error('no floor predictions found')
        rows = []
        for path in paths:
            pred = json.loads(path.read_text()); gt = load_floor_ground_truth(args.dataset, pred['floor_uid'])
            rows.append(evaluate_floor(pred, gt, FloorMetricProtocol(scale_track=args.scale_track)))
        result = write_metrics(rows, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
