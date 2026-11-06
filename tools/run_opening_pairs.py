#!/usr/bin/env python3
"""Build weak labels, train, calibrate, or infer opening pairs independently."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import torch
from portallayout.utils.frozen_opening_cache import FrozenOpeningCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('build-labels', 'train', 'match', 'calibrate'):
        p = sub.add_parser(name)
        p.add_argument('--prediction-cache', required=True); p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--torch-threads', type=int, default=2)
        if name == 'build-labels':
            p.add_argument('--dataset', required=True); p.add_argument('--review-file')
        if name in ('train', 'calibrate'): p.add_argument('--supervision', required=True)
        if name in ('match', 'calibrate'): p.add_argument('--checkpoint', required=True)
        if name == 'train':
            p.add_argument('--steps', type=int, default=200); p.add_argument('--batch-size', type=int, default=2)
            p.add_argument('--seed', type=int, default=123); p.add_argument('--resume')
            p.add_argument('--encoder-checkpoint'); p.add_argument('--training-mode', choices=('smoke', 'full'), default='smoke')
        if name == 'match': p.add_argument('--max-visualizations', type=int, default=12)
        if name == 'calibrate': p.add_argument('--min-precision', type=float, default=.95)
    args = parser.parse_args(); torch.set_num_threads(args.torch_threads)
    cache = FrozenOpeningCache(args.prediction_cache)
    if args.command == 'build-labels':
        from portallayout.datasets.opening_pair_supervision import build_weak_supervision
        result = build_weak_supervision(cache, args.dataset, args.output, args.review_file)['statistics']
    elif args.command == 'train':
        from portallayout.engine.opening_pair_training import train_pairs
        result = train_pairs(cache, args.supervision, args.output, steps=args.steps, batch_size=args.batch_size,
            seed=args.seed, device=args.device, resume=args.resume, training_mode=args.training_mode,
            encoder_checkpoint=args.encoder_checkpoint)
    elif args.command == 'match':
        from portallayout.engine.opening_pair_pipeline import match_floors
        result = match_floors(cache, args.checkpoint, args.output, device=args.device,
                              max_visualizations=args.max_visualizations)
    else:
        from portallayout.engine.opening_pair_training import calibrate_pairs
        result = calibrate_pairs(cache, args.supervision, args.checkpoint, args.output, args.device, args.min_precision)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
