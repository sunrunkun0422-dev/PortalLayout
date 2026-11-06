#!/usr/bin/env python3
"""Audit source integrity and independent supervision support before training."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from portallayout.datasets.floor_auxiliary_supervision import house_protocol,house_role
from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
from portallayout.engine.opening_pair_training import read_supervision
from portallayout.utils.room_feature_cache import write_json_atomic


def audit_pairs(cache, labels, protocol, minimum_houses=None):
    minimum_houses=minimum_houses or {'fit':10,'selection':5,'calibration':5}
    stats={};issues=[]
    for role in ('fit','selection','calibration'):
        rows=[r for r in labels['pairs'] if house_role(r['house_id'],protocol)==role]
        positives=[r for r in rows if any(t>=0 for t in r['target_a']+r['target_b'])]
        negatives=[r for r in rows if any(t==-1 for t in r['target_a']+r['target_b'])]
        active=[r for r in rows if any(t!=-2 for t in r['target_a']+r['target_b'])]
        houses=sorted({r['house_id'] for r in active})
        stats[role]=dict(pairs=len(rows),supervised_pairs=len(active),positive_pairs=len(positives),
                         proven_unmatched_pairs=len(negatives),supervised_houses=houses,
                         minimum_houses=minimum_houses[role])
        if len(houses)<minimum_houses[role]:issues.append(role+': insufficient supervised houses')
        if not positives or not negatives:issues.append(role+': missing positive or unmatched supervision')
    if any(house_role(r['house_id'],protocol)=='heldout' for r in labels['pairs']):
        issues.append('heldout houses present in training cache')
    return dict(status='ready_for_training' if not issues else 'insufficient_training_support',
                statistics=stats,issues=issues,source_files_verified=cache.check_unchanged(),
                note='Readiness checks data support, not future calibration success; unknown labels remain unknown.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-cache',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--supervision',required=True);p.add_argument('--output',required=True)
    p.add_argument('--minimum-fit-houses',type=int,default=10)
    p.add_argument('--minimum-selection-houses',type=int,default=5)
    p.add_argument('--minimum-calibration-houses',type=int,default=5)
    args=p.parse_args(argv)
    output=Path(args.output)
    if output.exists():p.error('output must be new')
    cache=FrozenOpeningCache(args.prediction_cache);labels=read_supervision(args.supervision,cache)
    limits={role:getattr(args,'minimum_'+role+'_houses') for role in ('fit','selection','calibration')}
    if min(limits.values())<1:p.error('house counts must be positive')
    report=audit_pairs(cache,labels,house_protocol(args.dataset),limits)
    write_json_atomic(output,report);print(json.dumps(report,indent=2))
    return 0 if report['status']=='ready_for_training' else 2

if __name__=='__main__':raise SystemExit(main())
