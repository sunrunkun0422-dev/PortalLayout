#!/usr/bin/env python3
"""Cache the fixed R01 cohort with RGB-only inputs and resumable identities."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portallayout.datasets.zind_floor_dataset import safe_path, validate_public_manifest
from portallayout.engine.floor_observation_pipeline import (BiLayoutObservationEncoder,
    ObservationPredictionSettings, frontend_contract, predict_floor_observations)
from portallayout.utils.room_feature_cache import (ObservationFeatureCache, content_hash,
                                                    fingerprint, write_json_atomic)


def selected_manifests(cohort, dataset_root, scope):
    allowed = {'fit', 'selection', 'calibration'} if scope == 'development' else {'audit'}
    roles = cohort['roles']
    all_houses = [house for values in roles.values() for house in values]
    if len(all_houses) != len(set(all_houses)):
        raise ValueError('cohort house roles overlap or contain duplicates')
    rows = [r for r in cohort['cohort'] if r['role'] in allowed]
    if not rows or len({r['floor_uid'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate floor cohort')
    manifests = []
    for row in rows:
        if row['house_id'] not in roles[row['role']]:
            raise ValueError('cohort row and role membership differ')
        path = safe_path(dataset_root, row['input_manifest'])
        if content_hash(path) != cohort['input_manifest_sha256'][row['floor_uid']]:
            raise ValueError('input manifest changed: '+str(path))
        manifest = validate_public_manifest(json.loads(path.read_text()))
        if manifest['floor_uid'] != row['floor_uid'] or len(manifest['observations']) != row['observations']:
            raise ValueError('input manifest does not match cohort')
        manifests.append(manifest)
    return manifests


def rgb_hashes(manifests, raw_root):
    paths = sorted({o['rgb']['relative_path'] for m in manifests for o in m['observations']})
    return {path: content_hash(safe_path(raw_root, path)) for path in paths}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--config', default=str(ROOT/'configs/research1_floor_frontend.json'))
    p.add_argument('--scope', choices=('development', 'audit'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error('batch size must be positive')
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset_root, raw_root = Path(args.dataset).resolve(), Path(args.raw_root).resolve()
    cohort_path = Path(args.cohort).resolve()
    cohort = json.loads(cohort_path.read_text())
    if Path(cohort['dataset_root']).resolve() != dataset_root or Path(cohort['raw_root']).resolve() != raw_root:
        raise ValueError('dataset/raw roots differ from the frozen cohort')
    manifests = selected_manifests(cohort, dataset_root, args.scope)
    cfg = json.loads(Path(args.config).read_text())
    settings = ObservationPredictionSettings(**cfg['prediction'])
    paths = [(ROOT/cfg[k]).resolve() for k in ('bi_layout_config', 'bi_layout_checkpoint', 'opening_checkpoint')]
    contract = frontend_contract(*paths, settings, execution_device=args.device)
    contract['cohort_sha256'] = content_hash(cohort_path)
    cache = ObservationFeatureCache(args.output, contract, resume=args.resume)
    source_file = cache.root/'rgb_source_sha256.json'
    inventory = {'schema_version': 'research10.rgb-source.v1', 'cohort_sha256': content_hash(cohort_path),
                 'scope': args.scope, 'images': rgb_hashes(manifests, raw_root)}
    inventory['fingerprint'] = fingerprint(inventory)
    if source_file.exists():
        if json.loads(source_file.read_text()) != inventory:
            raise ValueError('RGB sources changed since cache creation')
    else:
        if (cache.root/'observations').exists():
            raise ValueError('missing RGB inventory for existing observations')
        write_json_atomic(source_file, inventory)
    print(json.dumps({'stage': 'rgb_inventory', 'images': len(inventory['images']),
                      'floors': len(manifests), 'scope': args.scope}), flush=True)
    encoder = BiLayoutObservationEncoder.from_checkpoints(*paths, settings, args.device)
    write_json_atomic(cache.root/'model_report.json', encoder.report)
    floors, inferred, reused = [], 0, 0
    for index, manifest in enumerate(manifests):
        result = predict_floor_observations([manifest], raw_root, encoder, cache, settings,
                                            args.batch_size, write_report=False)
        floors.extend(result['floors'])
        inferred += result['observations_inferred']
        reused += result['observations_reused']
        if result['status'] != 'complete':
            raise ValueError('input errors at floor '+manifest['floor_uid'])
        progress = {'status': 'running', 'completed_floors': index+1, 'total_floors': len(manifests),
                    'observations_inferred': inferred, 'observations_reused': reused,
                    'last_floor_uid': manifest['floor_uid']}
        write_json_atomic(cache.root/'progress.json', progress)
        print(json.dumps(progress), flush=True)
    if rgb_hashes(manifests, raw_root) != inventory['images']:
        raise ValueError('RGB source changed during caching')
    report = {'status': 'complete', 'floor_samples': len(manifests), 'floors': floors,
              'unique_observations': len({o['observation_id'] for m in manifests for o in m['observations']}),
              'observations_inferred': inferred,
              'observations_reused': reused, 'input_errors': [], 'ground_truth_read': False,
              'contract_fingerprint': cache.fingerprint, 'scope': args.scope,
              'cohort_sha256': content_hash(cohort_path), 'rgb_inventory_sha256': content_hash(source_file),
              'pretraining_exposure': 'unknown'}
    write_json_atomic(cache.root/'run_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
