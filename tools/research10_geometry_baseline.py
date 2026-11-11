#!/usr/bin/env python3
"""R02/R03 diagnostic: current synthetic geometry and frozen paired evidence.

This script does not train or evaluate the new R01 cohort. Historical GT is
used only in already-frozen, offline diagnostic artifacts.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portallayout.evaluation.pose_candidate_audit import synthetic_coordinate_audit
from portallayout.utils.room_feature_cache import content_hash, write_json_atomic


def recompute_paired(path):
    groups = defaultdict(list)
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            groups[row['collection']].append(row)
    variants = ('dominant_wall_s1p0', 'dominant_wall_s0p0',
                'local_m2_s1p0', 'local_m2_s0p0')
    output = {}
    for group, rows in sorted(groups.items()):
        identities = [(r['house_id'], r['pair_id']) for r in rows]
        if len(identities) != len(set(identities)):
            raise ValueError('duplicate fixed pair in group '+group)
        table = {}
        for name in variants:
            houses = defaultdict(list)
            for row in rows:
                if row[name] not in ('0', '1'):
                    raise ValueError('non-binary frozen candidate outcome')
                houses[row['house_id']].append(int(row[name]))
            house_rates = {h: float(np.mean(v)) for h, v in sorted(houses.items())}
            table[name] = dict(success=sum(sum(v) for v in houses.values()), total=len(rows),
                               houses=len(houses), house_macro=float(np.mean(list(house_rates.values()))),
                               house_rates=house_rates)
        comparisons = {}
        for baseline, variant in ((variants[0], variants[3]), (variants[0], variants[1]),
                                  (variants[1], variants[3]), (variants[0], variants[2])):
            a, b = table[baseline]['house_rates'], table[variant]['house_rates']
            if set(a) != set(b):
                raise ValueError('paired house support mismatch')
            delta = np.array([b[h]-a[h] for h in sorted(a)])
            rng = np.random.RandomState(123)
            boot = delta[rng.randint(0, len(delta), size=(2000, len(delta)))].mean(1)
            comparisons[baseline+'__to__'+variant] = {
                'house_macro_delta': float(delta.mean()), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist()}
        output[group] = {'variants': table, 'comparisons': comparisons}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new/empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT/'docs/method_advantage_evidence_20261028/geometry_paired_selection.csv'
    edge_path = ROOT/'experiments/research1_pose_geometry_20261026_0245/edge_error_summary.json'
    coordinate_path = ROOT/'experiments/research1_pose_geometry_20261026_0245/coordinate_audit.json'
    synthetic = synthetic_coordinate_audit()
    historical = json.loads(coordinate_path.read_text())
    pairs = recompute_paired(csv_path)
    edges = json.loads(edge_path.read_text())
    report = dict(schema_version='research10.r02-r03-diagnostic.v1',
                  status='partial_historical_diagnostic_only', synthetic_current=synthetic,
                  historical_real_coordinate_checks={
                      'floors': len(historical['real_floors']),
                      'checks': sum(r['checks'] for r in historical['real_floors']),
                      'projection_max_error': max(r['projection_max_error'] for r in historical['real_floors']),
                      'angle_max_error': max(r['angle_max_error'] for r in historical['real_floors'])},
                  fixed_pairs=pairs, edge_diagnostics=edges,
                  source_sha256={str(p.relative_to(ROOT)): content_hash(p) for p in (csv_path, edge_path, coordinate_path)},
                  bootstrap={'unit': 'house', 'iterations': 2000, 'seed': 123},
                  limitations=['historical development data; not the new R01 cohort',
                               'candidate solvability is not selected edge accuracy',
                               'no inference, calibration, new training or human validation was run'])
    write_json_atomic(output/'diagnostic.json', report)
    print(json.dumps({'status': report['status'], 'synthetic_cases': len(synthetic['cases']),
                      'historical_coordinate_checks': report['historical_real_coordinate_checks'],
                      'r3_total': pairs['r3']['variants']['dominant_wall_s1p0']['total'],
                      'r3_baseline': pairs['r3']['variants']['dominant_wall_s1p0']['success'],
                      'r3_scale_only': pairs['r3']['variants']['dominant_wall_s0p0']['success'],
                      'r3_full': pairs['r3']['variants']['local_m2_s0p0']['success']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
