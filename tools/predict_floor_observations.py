#!/usr/bin/env python3
"""Predict/cache every RGB observation in selected official-train floor samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.datasets.zind_floor_dataset import ZInDFloorDataset
from portallayout.engine.floor_observation_pipeline import (
    BiLayoutObservationEncoder, ObservationPredictionSettings,
    frontend_contract, predict_floor_observations,
)
from portallayout.utils.room_feature_cache import ObservationFeatureCache, write_json_atomic


def select_inputs(dataset, house_ids=None, floor_uids=None, max_floors=None):
    rows = [(i, row) for i, row in enumerate(dataset.samples)
            if (not house_ids or row['house_id'] in house_ids) and
               (not floor_uids or row['floor_uid'] in floor_uids)]
    if floor_uids and set(floor_uids) - {row['floor_uid'] for _, row in rows}:
        raise ValueError('requested floor UID is absent from the selected train subset')
    if house_ids and set(house_ids) - {row['house_id'] for _, row in rows}:
        raise ValueError('requested house is absent from the selected train subset')
    if max_floors is not None:
        if max_floors < 1:
            raise ValueError('max floors must be positive')
        rows = rows[:max_floors]
    if not rows:
        raise ValueError('no floor samples selected')
    return [dataset[index] for index, _ in rows]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--config', default=str(REPO_ROOT / 'configs/research1_floor_frontend.json'))
    p.add_argument('--output')
    p.add_argument('--house-ids', nargs='+')
    p.add_argument('--floor-uids', nargs='+')
    p.add_argument('--training-subset', choices=('fit', 'dev'))
    p.add_argument('--sample-view', choices=('all_panoramas', 'primary_only', 'one_per_space', 'sampled_N_views'),
                   default='all_panoramas')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--max-floors', type=int)
    group.add_argument('--all-floors', action='store_true')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='auto')
    p.add_argument('--torch-threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.torch_threads < 1:
        p.error('batch size and torch threads must be positive')
    cfg = json.loads(Path(args.config).read_text())
    settings = ObservationPredictionSettings(**cfg['prediction'])
    dataset = ZInDFloorDataset(args.dataset, sample_view=args.sample_view,
                              official_split='train', training_subset=args.training_subset)
    limit = args.max_floors
    if limit is None and not args.all_floors and not args.floor_uids:
        limit = 1  # Full cache generation requires an explicit scope.
    manifests = select_inputs(dataset, args.house_ids, args.floor_uids, limit)
    summary = {'floor_samples': len(manifests), 'official_split': 'train',
               'unique_observations': len({x['observation_id'] for m in manifests for x in m['observations']}),
               'sample_ids': [m['sample_id'] for m in manifests],
               'floor_uids': [m['floor_uid'] for m in manifests], 'ground_truth_read': False}
    if args.dry_run:
        print(json.dumps(dict(summary, model_loaded=False, image_decode_run=False), indent=2))
        return 0
    if not args.output:
        p.error('--output is required for prediction')
    import torch
    import numpy as np
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(123)
    np.random.seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    paths = {key: (REPO_ROOT / cfg[key]).resolve() for key in
             ('bi_layout_config', 'bi_layout_checkpoint', 'opening_checkpoint')}
    print(json.dumps(dict(summary, device=device)), flush=True)
    contract = frontend_contract(*paths.values(), settings, execution_device=device)
    cache = ObservationFeatureCache(args.output, contract, resume=args.resume)
    encoder = BiLayoutObservationEncoder.from_checkpoints(*paths.values(), settings, device)
    write_json_atomic(cache.root / 'model_report.json', encoder.report)
    report = predict_floor_observations(manifests, args.raw_root, encoder, cache, settings, args.batch_size)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
